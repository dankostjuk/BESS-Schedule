import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

// ─── Serialisable types ───────────────────────────────────────────────────────

@Serializable
data class PointRecord(
    val ts: String,
    val yTrue: Double,
    val yPredHgbr: Double,
    val yPredCatboost: Double,
    val yPredXgboost: Double
)

@Serializable
data class BessEvalRequest(val file: String)

@Serializable
data class ErrorResponse(val error: String)

// ─── CSV loader ───────────────────────────────────────────────────────────────

fun loadPointCsv(path: String): List<PointRecord> {
    val file = File(path)
    if (!file.exists()) { System.err.println("CSV not found: ${file.absolutePath}"); return emptyList() }
    return file.readLines().drop(1).mapNotNull { line ->
        runCatching {
            val p = line.split(",")
            PointRecord(
                ts            = p[0].trim().substring(0, 16),
                yTrue         = p[1].trim().toDouble(),
                yPredHgbr     = p[2].trim().toDouble(),
                yPredCatboost = p[3].trim().toDouble(),
                yPredXgboost  = p[4].trim().toDouble()
            )
        }.getOrNull()
    }
}

// ─── Python process helper ────────────────────────────────────────────────────

data class ProcResult(val stdout: String, val stderr: String, val exitCode: Int)

suspend fun runPython(vararg args: String): ProcResult = withContext(Dispatchers.IO) {
    val python = "../market_ui/.venv-mapie/bin/python"
    val process = ProcessBuilder(python, *args)
        .directory(File("."))
        .start()

    val outBuf = StringBuilder()
    val errBuf = StringBuilder()

    val t1 = Thread { outBuf.append(process.inputStream.bufferedReader().readText()) }.also { it.start() }
    val t2 = Thread { errBuf.append(process.errorStream.bufferedReader().readText()) }.also { it.start() }

    val code = process.waitFor()
    t1.join(); t2.join()

    ProcResult(outBuf.toString(), errBuf.toString(), code)
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fun main() {
    val pointData = loadPointCsv("data/point_prediction.csv")
    println("Loaded ${pointData.size} point-prediction records")

    val indexHtml = ClassLoader.getSystemResourceAsStream("static/index.html")
        ?.bufferedReader()?.readText()
        ?: error("static/index.html not found in classpath")

    embeddedServer(Netty, port = 8080) {
        install(ContentNegotiation) {
            json(Json { prettyPrint = false })
        }

        routing {

            // ── Frontend ────────────────────────────────────────────────────
            get("/") {
                call.respondText(indexHtml, ContentType.Text.Html)
            }

            // ── Point prediction data ───────────────────────────────────────
            get("/api/data") {
                call.respond(pointData)
            }

            // ── List interval-prediction CSV files ──────────────────────────
            get("/api/interval-files") {
                val files = File("data").listFiles()
                    ?.filter { it.isFile && it.name.contains("interval") && it.name.endsWith(".csv") }
                    ?.map { it.name }
                    ?.sorted()
                    ?: emptyList()
                call.respond(files)
            }

            // ── Run BESS evaluation via Python ──────────────────────────────
            post("/api/bess-eval") {
                val req = call.receive<BessEvalRequest>()
                val filename = req.file

                // Guard against path traversal
                if (filename.contains('/') || filename.contains("..") || !filename.endsWith(".csv")) {
                    call.respond(HttpStatusCode.BadRequest, ErrorResponse("Invalid filename"))
                    return@post
                }

                val csvPath = "data/$filename"
                if (!File(csvPath).exists()) {
                    call.respond(HttpStatusCode.NotFound, ErrorResponse("File not found: $filename"))
                    return@post
                }

                val result = runPython("scripts/bess.py", csvPath)

                if (result.exitCode != 0) {
                    val msg = result.stderr.ifBlank { result.stdout }.take(500)
                    call.respond(HttpStatusCode.InternalServerError, ErrorResponse("Python error: $msg"))
                    return@post
                }

                // Forward raw JSON produced by the Python script
                call.respondText(result.stdout.trim(), ContentType.Application.Json)
            }
        }
    }.start(wait = true)

    println("Working directory : ${File(".").absolutePath}")
    println("Server running at : http://localhost:8080")
}
