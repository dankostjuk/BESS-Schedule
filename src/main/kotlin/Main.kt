import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

// ─── Types ────────────────────────────────────────────────────────────────────

@Serializable data class ModelFile(val file: String, val name: String)

@Serializable
data class SchedulePoint(
    val ts: String,
    val yPred: Double,
    val piLo: Double? = null,
    val piHi: Double? = null,
    val charge: Double,
    val discharge: Double,
    val soc: Double,
    val truePrice: Double,
    val chargeTrue: Double,
    val dischargeTrue: Double,
    val socTrue: Double,
    val chargePeak: Double,
    val dischargePeak: Double,
    val socPeak: Double,
    val chargeYday: Double,
    val dischargeYday: Double,
    val socYday: Double,
)

@Serializable
data class DailyProfit(
    val date: String,
    val predProfit: Double,
    val trueProfit: Double,
    val peakProfit: Double,
    val ydayProfit: Double,
)

@Serializable
data class ScheduleResponse(
    val points: List<SchedulePoint>,
    val dailyProfits: List<DailyProfit>,
    val hasInterval: Boolean,
)

@Serializable data class TruePricePoint(val ts: String, val price: Double)

@Serializable data class ErrorResponse(val error: String)

// ─── Helpers ─────────────────────────────────────────────────────────────────

fun parseModelName(filename: String): String = when {
    filename.startsWith("pto_")    -> filename.removePrefix("pto_").removeSuffix(".csv")
    filename.startsWith("robust_") -> filename.removePrefix("robust_").removeSuffix(".csv")
    else                           -> filename.removeSuffix(".csv")
}

fun listModelFiles(): List<ModelFile> =
    File("data").listFiles()
        ?.filter { it.isFile && it.name.endsWith(".csv") &&
                   (it.name.startsWith("pto_") || it.name.startsWith("robust_")) }
        ?.map { ModelFile(file = it.name, name = parseModelName(it.name)) }
        ?.sortedBy { it.name }
        ?: emptyList()

fun readCsv(path: String): Pair<List<String>, List<List<String>>> {
    val lines = File(path).readLines()
    if (lines.isEmpty()) return emptyList<String>() to emptyList()
    return lines[0].split(",") to lines.drop(1).map { it.split(",") }
}

fun loadTruePrices(): List<TruePricePoint> {
    // Use CEST timestamps from the first available pto_ file so the frontend
    // sees the same timezone as model data (+02:00), then pair with true prices.
    val modelFile = File("data").listFiles()
        ?.filter { it.isFile && it.name.startsWith("pto_") && it.name.endsWith(".csv") }
        ?.minByOrNull { it.name } ?: return emptyList()

    val (_, mRows) = readCsv("data/${modelFile.name}")
    val (tHead, tRows) = readCsv("data/schedule_true.csv")
    val tPrice = tHead.indexOf("15min cena (EUR/MWh)")

    return mRows.zip(tRows).mapNotNull { (m, t) ->
        val ts    = m[0].trim()
        val price = t.getOrNull(tPrice)?.trim()?.toDoubleOrNull() ?: return@mapNotNull null
        TruePricePoint(ts = ts, price = price)
    }
}

fun loadSchedule(filename: String): ScheduleResponse {
    val hasInterval = filename.startsWith("robust_")

    val (mHead, mRows) = readCsv("data/$filename")
    val (tHead, tRows) = readCsv("data/schedule_true.csv")
    val (pHead, pRows) = readCsv("data/schedule_peak_hours.csv")
    val (yHead, yRows) = readCsv("data/schedule_yesterday_hours.csv")

    fun col(h: List<String>, name: String) = h.indexOf(name)
    fun List<String>.d(i: Int) = getOrNull(i)?.trim()?.toDoubleOrNull()

    // model file columns
    val mYPred  = col(mHead, "y_pred")
    val mPiLo   = col(mHead, "pi_lo")
    val mPiHi   = col(mHead, "pi_hi")
    val mProfit = col(mHead, "0")
    val mCharge = col(mHead, "charge")
    val mDisch  = col(mHead, "discharge")
    val mSoc    = col(mHead, "soc_end")

    // schedule_true columns
    val tPrice  = col(tHead, "15min cena (EUR/MWh)")
    val tProfit = col(tHead, "0")
    val tCharge = col(tHead, "charge")
    val tDisch  = col(tHead, "discharge")
    val tSoc    = col(tHead, "soc_end")

    // schedule_peak_hours columns (ts is explicit header col 0)
    val pProfit = col(pHead, "0")
    val pCharge = col(pHead, "charge")
    val pDisch  = col(pHead, "discharge")
    val pSoc    = col(pHead, "soc_end")

    // schedule_yesterday_hours columns
    val yProfit = col(yHead, "0")
    val yCharge = col(yHead, "charge")
    val yDisch  = col(yHead, "discharge")
    val ySoc    = col(yHead, "soc_end")

    val points       = mutableListOf<SchedulePoint>()
    val dailyProfits = mutableListOf<DailyProfit>()
    val seenDates    = mutableSetOf<String>()

    val n = listOf(mRows.size, tRows.size, pRows.size, yRows.size).min()
    for (i in 0 until n) {
        val m = mRows[i]; val t = tRows[i]; val p = pRows[i]; val y = yRows[i]
        val ts   = m[0].trim()
        val date = ts.substring(0, 10)

        val yPred     = m.d(mYPred)    ?: continue
        val truePrice = t.d(tPrice)    ?: continue

        points.add(SchedulePoint(
            ts            = ts,
            yPred         = yPred,
            piLo          = if (hasInterval) m.d(mPiLo) else null,
            piHi          = if (hasInterval) m.d(mPiHi) else null,
            charge        = m.d(mCharge) ?: 0.0,
            discharge     = m.d(mDisch)  ?: 0.0,
            soc           = m.d(mSoc)    ?: 0.0,
            truePrice     = truePrice,
            chargeTrue    = t.d(tCharge) ?: 0.0,
            dischargeTrue = t.d(tDisch)  ?: 0.0,
            socTrue       = t.d(tSoc)    ?: 0.0,
            chargePeak    = p.d(pCharge) ?: 0.0,
            dischargePeak = p.d(pDisch)  ?: 0.0,
            socPeak       = p.d(pSoc)    ?: 0.0,
            chargeYday    = y.d(yCharge) ?: 0.0,
            dischargeYday = y.d(yDisch)  ?: 0.0,
            socYday       = y.d(ySoc)    ?: 0.0,
        ))

        if (date !in seenDates) {
            val mP = m.d(mProfit); val tP = t.d(tProfit)
            if (mP != null && tP != null) {
                seenDates += date
                dailyProfits += DailyProfit(
                    date       = date,
                    predProfit = mP,
                    trueProfit = tP,
                    peakProfit = p.d(pProfit) ?: 0.0,
                    ydayProfit = y.d(yProfit) ?: 0.0,
                )
            }
        }
    }

    return ScheduleResponse(points = points, dailyProfits = dailyProfits, hasInterval = hasInterval)
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fun main() {
    val indexHtml = ClassLoader.getSystemResourceAsStream("static/index.html")
        ?.bufferedReader()?.readText()
        ?: error("static/index.html not found in classpath")

    embeddedServer(Netty, port = 8080) {
        install(ContentNegotiation) { json(Json { prettyPrint = false }) }

        routing {
            get("/") { call.respondText(indexHtml, ContentType.Text.Html) }

            get("/api/files") { call.respond(listModelFiles()) }

            get("/api/true-prices") { call.respond(loadTruePrices()) }

            get("/api/schedule-data") {
                val filename = call.request.queryParameters["file"] ?: run {
                    call.respond(HttpStatusCode.BadRequest, ErrorResponse("Missing file parameter"))
                    return@get
                }
                if (filename.contains('/') || filename.contains("..") || !filename.endsWith(".csv")) {
                    call.respond(HttpStatusCode.BadRequest, ErrorResponse("Invalid filename"))
                    return@get
                }
                if (!File("data/$filename").exists()) {
                    call.respond(HttpStatusCode.NotFound, ErrorResponse("File not found: $filename"))
                    return@get
                }
                call.respond(loadSchedule(filename))
            }
        }
    }.start(wait = true)

    println("Working directory : ${File(".").absolutePath}")
    println("Server running at : http://localhost:8080")
}
