import java.awt.*
import java.awt.event.*
import java.awt.geom.GeneralPath
import java.io.File
import java.time.*
import java.time.format.DateTimeFormatter
import java.util.Date
import javax.swing.*
import javax.swing.border.EmptyBorder
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

// ------------------------------------------------------------
// Domain model
// ------------------------------------------------------------

interface TemporalPoint {
    val ts: OffsetDateTime
}

data class DataPoint(
    override val ts: OffsetDateTime,
    val yTrue: Double,
    val predictions: Map<String, Double>
) : TemporalPoint

data class MethodConfig(
    val key: String,
    val label: String,
    val color: Color,
    var visible: Boolean = true
)

data class Metrics(val mae: Double, val rmse: Double, val mape: Double)

data class BessSchedulePoint(
    override val ts: OffsetDateTime,
    val chargePred: Double,
    val dischargePred: Double,
    val chargeTrue: Double,
    val dischargeTrue: Double
) : TemporalPoint {
    val netPred: Double get() = chargePred - dischargePred
    val netTrue: Double get() = chargeTrue - dischargeTrue
}

data class DailyProfitRow(
    val date: LocalDate,
    val predProfit: Double,
    val trueProfit: Double
)

data class DailyProfitPoint(
    val date: LocalDate,
    val cumPredProfit: Double,
    val cumTrueProfit: Double
) : TemporalPoint {
    override val ts: OffsetDateTime = date.atStartOfDay().atOffset(ZoneOffset.UTC)
    val lostProfit: Double get() = cumTrueProfit - cumPredProfit
}

data class BessSummary(
    val predProfit: Double,
    val trueProfit: Double,
    val lostProfit: Double,
    val lostProfitAvg: Double,
    val chargeMatch: Double,
    val dischargeMatch: Double,
    val daysEvaluated: Int
)

data class SeriesDef<T : TemporalPoint>(
    val label: String,
    val color: Color,
    val valueOf: (T) -> Double,
    val isVisible: () -> Boolean = { true },
    val dashed: Boolean = false,
    val strokeWidth: Float = 1.6f
)

// ------------------------------------------------------------
// Calculations
// ------------------------------------------------------------
fun computeWape(trueVals: List<Double>, predVals: List<Double>): Double {
    if (trueVals.isEmpty()) return 0.0

    val numerator = trueVals.zip(predVals).sumOf { (t, p) -> abs(t - p) }
    val denominator = trueVals.sumOf { abs(it) }

    return if (denominator < 1e-12) 0.0 else numerator / denominator * 100.0
}

fun computeMetrics(trueVals: List<Double>, predVals: List<Double>): Metrics {
    if (trueVals.isEmpty()) return Metrics(0.0, 0.0, 0.0)

    val pairs = trueVals.zip(predVals)
    val n = pairs.size.toDouble()

    val mae = pairs.sumOf { (t, p) -> abs(t - p) } / n
    val rmse = sqrt(pairs.sumOf { (t, p) -> (t - p).pow(2) } / n)
    val wape = computeWape(trueVals, predVals)

    return Metrics(mae, rmse, wape)
}

fun computeBessSummary(schedule: List<BessSchedulePoint>, daily: List<DailyProfitPoint>): BessSummary {
    val predProfit = daily.lastOrNull()?.cumPredProfit ?: 0.0
    val trueProfit = daily.lastOrNull()?.cumTrueProfit ?: 0.0
    val lostProfit = trueProfit - predProfit
    val days = daily.size.coerceAtLeast(1)

    fun matchRatio(pred: (BessSchedulePoint) -> Double, truth: (BessSchedulePoint) -> Double): Double {
        if (schedule.isEmpty()) return 0.0
        val denom = schedule.count { pred(it) > 1e-9 }
        if (denom == 0) return 0.0
        val matches = schedule.count { pred(it) > 1e-9 && truth(it) > 1e-9 }
        return matches.toDouble() / denom.toDouble()
    }

    return BessSummary(
        predProfit = predProfit,
        trueProfit = trueProfit,
        lostProfit = lostProfit,
        lostProfitAvg = lostProfit / days,
        chargeMatch = matchRatio({ it.chargePred }, { it.chargeTrue }),
        dischargeMatch = matchRatio({ it.dischargePred }, { it.dischargeTrue }),
        daysEvaluated = daily.size
    )
}

// ------------------------------------------------------------
// CSV loading (header-based and Kotlin-first)
// ------------------------------------------------------------

private val offsetDateFormats = listOf(
    DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ssxxx"),
    DateTimeFormatter.ISO_OFFSET_DATE_TIME,
    DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ssX")
)

private fun parseOffsetDateTime(raw: String): OffsetDateTime {
    val normalized = raw.trim()
    offsetDateFormats.forEach { fmt ->
        runCatching { return OffsetDateTime.parse(normalized, fmt) }
    }
    return OffsetDateTime.parse(normalized)
}

private fun parseCsvRows(path: String): List<Map<String, String>> {
    val file = File(path)
    if (!file.exists()) return emptyList()
    val lines = file.readLines().filter { it.isNotBlank() }
    if (lines.size < 2) return emptyList()

    val headers = lines.first().split(',').map { it.trim() }
    return lines.drop(1).mapNotNull { line ->
        val parts = line.split(',').map { it.trim() }
        if (parts.size < headers.size) return@mapNotNull null
        headers.indices.associate { idx -> headers[idx] to parts[idx] }
    }
}

private fun Map<String, String>.valueFor(vararg keys: String): String? =
    keys.firstNotNullOfOrNull { key -> this[key] ?: this[key.lowercase()] ?: this[key.uppercase()] }

private fun Map<String, String>.doubleFor(vararg keys: String): Double =
    valueFor(*keys)?.toDoubleOrNull() ?: 0.0

fun loadPointCsv(path: String): List<DataPoint> =
    parseCsvRows(path).mapNotNull { row ->
        runCatching {
            DataPoint(
                ts = parseOffsetDateTime(row.valueFor("ts", "timestamp") ?: error("missing ts")),
                yTrue = row.doubleFor("yTrue", "y_true", "actual"),
                predictions = mapOf(
                    "hgbr" to row.doubleFor("hgbr", "yPredHgbr", "y_pred_hgbr"),
                    "catboost" to row.doubleFor("catboost", "yPredCatboost", "y_pred_catboost"),
                    "xgboost" to row.doubleFor("xgboost", "yPredXgboost", "y_pred_xgboost")
                )
            )
        }.getOrNull()
    }.sortedBy { it.ts.toInstant() }

fun loadBessScheduleCsv(path: String): List<BessSchedulePoint> =
    parseCsvRows(path).mapNotNull { row ->
        runCatching {
            BessSchedulePoint(
                ts = parseOffsetDateTime(row.valueFor("ts", "timestamp") ?: error("missing ts")),
                chargePred = row.doubleFor("chargePred", "charge_pred"),
                dischargePred = row.doubleFor("dischargePred", "discharge_pred"),
                chargeTrue = row.doubleFor("chargeTrue", "charge_true"),
                dischargeTrue = row.doubleFor("dischargeTrue", "discharge_true")
            )
        }.getOrNull()
    }.sortedBy { it.ts.toInstant() }

fun loadDailyProfitCsv(path: String): List<DailyProfitRow> =
    parseCsvRows(path).mapNotNull { row ->
        runCatching {
            DailyProfitRow(
                date = LocalDate.parse(row.valueFor("date") ?: error("missing date")),
                predProfit = row.doubleFor("predProfit", "pred_profit"),
                trueProfit = row.doubleFor("trueProfit", "true_profit")
            )
        }.getOrNull()
    }.sortedBy { it.date }

fun buildProfitPointsForRange(
    rows: List<DailyProfitRow>,
    start: LocalDate,
    end: LocalDate
): List<DailyProfitPoint> {
    var cumPred = 0.0
    var cumTrue = 0.0

    return rows.asSequence()
        .filter { it.date in start..end }
        .map { row ->
            cumPred += row.predProfit
            cumTrue += row.trueProfit
            DailyProfitPoint(
                date = row.date,
                cumPredProfit = cumPred,
                cumTrueProfit = cumTrue
            )
        }
        .toList()
}
// ------------------------------------------------------------
// Shared date-range model
// ------------------------------------------------------------

class SharedRangeModel(start: LocalDate, end: LocalDate) {
    private val listeners = mutableListOf<(LocalDate, LocalDate) -> Unit>()

    var start: LocalDate = start
        private set
    var end: LocalDate = end
        private set

    fun update(newStart: LocalDate, newEnd: LocalDate) {
        val normalizedStart = minOf(newStart, newEnd)
        val normalizedEnd = maxOf(newStart, newEnd)
        if (normalizedStart == start && normalizedEnd == end) return
        start = normalizedStart
        end = normalizedEnd
        listeners.forEach { it(start, end) }
    }

    fun shiftDays(days: Long, min: LocalDate, max: LocalDate) {
        val span = Duration.between(start.atStartOfDay(), end.plusDays(1).atStartOfDay()).toDays().coerceAtLeast(1)
        var newStart = start.plusDays(days)
        val tentativeEnd = end.plusDays(days)

        if (newStart < min) {
            val correction = Duration.between(newStart.atStartOfDay(), min.atStartOfDay()).toDays()
            newStart = newStart.plusDays(correction)
        }
        if (tentativeEnd > max) {
            val correction = Duration.between(max.atStartOfDay(), tentativeEnd.atStartOfDay()).toDays()
            newStart = newStart.minusDays(correction)
        }

        val adjustedEnd = if (newStart.plusDays(span - 1) <= max) newStart.plusDays(span - 1) else max
        update(newStart.coerceAtLeast(min), adjustedEnd.coerceAtMost(max))
    }

    fun onChange(listener: (LocalDate, LocalDate) -> Unit) {
        listeners += listener
    }
}

// ------------------------------------------------------------
// Generic chart panel
// ------------------------------------------------------------

class MultiSeriesTimeChart<T : TemporalPoint>(
    private var source: List<T>,
    private val seriesDefs: List<SeriesDef<T>>,
    private val yAxisTitle: String,
    private val tooltipLines: (T) -> List<String>,
    private val anchorEnabled: Boolean = false
) : JPanel() {

    private val pad = Insets(48, 80, 56, 24)
    private val anchorColor = Color(255, 210, 60)
    private val axisFmt = DateTimeFormatter.ofPattern("MM-dd HH:mm")
    private val anchorFmt = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")

    private var dragAnchor: Point? = null
    private var dragViewStart = 0
    private var wasDragged = false
    private var tooltipText: String? = null
    private var tooltipPos: Point? = null

    var filtered: List<T> = source
        private set

    var viewStart = 0
        private set
    var viewEnd = (source.lastIndex).coerceAtLeast(0)
        private set
    var selectedIdx: Int? = null
        private set

    var onViewChanged: ((List<T>) -> Unit)? = null
    var onPointSelected: ((T) -> Unit)? = null

    init {
        background = Color(22, 22, 30)
        cursor = Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR)

        addMouseWheelListener { e ->
            if (filtered.size < 2) return@addMouseWheelListener
            val range = (viewEnd - viewStart).coerceAtLeast(1)
            val chartW = (width - pad.left - pad.right).coerceAtLeast(1)
            val frac = ((e.x - pad.left).toDouble() / chartW).coerceIn(0.0, 1.0)
            val center = viewStart + frac * range
            val factor = if (e.wheelRotation > 0) 1.15 else 0.87
            val newRange = (range * factor).toInt().coerceIn(8, filtered.lastIndex.coerceAtLeast(1))
            val maxStart = (filtered.size - 1 - newRange).coerceAtLeast(0)
            viewStart = (center - frac * newRange).toInt().coerceIn(0, maxStart)
            viewEnd = (viewStart + newRange).coerceIn(viewStart, filtered.lastIndex)
            fireViewChanged()
            repaint()
        }

        addMouseListener(object : MouseAdapter() {
            override fun mousePressed(e: MouseEvent) {
                dragAnchor = e.point
                dragViewStart = viewStart
                wasDragged = false
            }

            override fun mouseReleased(e: MouseEvent) {
                if (!wasDragged && anchorEnabled) {
                    screenXToIdx(e.x)?.let { idx ->
                        selectedIdx = idx
                        onPointSelected?.invoke(filtered[idx])
                        repaint()
                    }
                }
                dragAnchor = null
            }
        })

        addMouseMotionListener(object : MouseMotionAdapter() {
            override fun mouseDragged(e: MouseEvent) {
                if (filtered.size < 2) return
                val anchor = dragAnchor ?: return
                wasDragged = true
                val range = (viewEnd - viewStart).coerceAtLeast(1)
                val chartW = (width - pad.left - pad.right).coerceAtLeast(1)
                val delta = ((anchor.x - e.x).toDouble() / chartW * range).toInt()
                val maxStart = (filtered.size - 1 - range).coerceAtLeast(0)
                viewStart = (dragViewStart + delta).coerceIn(0, maxStart)
                viewEnd = (viewStart + range).coerceIn(viewStart, filtered.lastIndex)
                fireViewChanged()
                repaint()
            }

            override fun mouseMoved(e: MouseEvent) {
                updateTooltip(e.point)
                repaint()
            }
        })
    }

    fun setDateFilter(start: LocalDate, end: LocalDate) {
        filtered = source.filter {
            val d = it.ts.toLocalDate()
            d >= start && d <= end
        }
        if (filtered.isEmpty()) {
            viewStart = 0
            viewEnd = 0
            selectedIdx = null
            tooltipText = null
            fireViewChanged()
            repaint()
            return
        }
        viewStart = 0
        viewEnd = filtered.lastIndex
        selectedIdx = selectedIdx?.takeIf { it in filtered.indices }
        fireViewChanged()
        repaint()
    }

    fun resetZoom() {
        if (filtered.isEmpty()) return
        viewStart = 0
        viewEnd = filtered.lastIndex
        fireViewChanged()
        repaint()
    }

    fun replaceSource(newSource: List<T>) {
        source = newSource
        filtered = source
        viewStart = 0
        viewEnd = source.lastIndex.coerceAtLeast(0)
        selectedIdx = null
        tooltipText = null
        tooltipPos = null
        fireViewChanged()
        repaint()
    }

    fun zoomFromSelected(duration: Duration) {
        val idx = selectedIdx ?: return
        if (filtered.isEmpty()) return
        val startTs = filtered[idx].ts
        val endTs = startTs.plus(duration)
        val endIdx = filtered.indexOfLast { it.ts <= endTs }.let { if (it < 0) filtered.lastIndex else it }
        viewStart = idx.coerceIn(0, filtered.lastIndex)
        viewEnd = endIdx.coerceIn(viewStart, filtered.lastIndex)
        fireViewChanged()
        repaint()
    }

    private fun fireViewChanged() {
        if (filtered.isEmpty()) onViewChanged?.invoke(emptyList())
        else onViewChanged?.invoke(filtered.subList(viewStart, viewEnd + 1))
    }

    private fun screenXToIdx(screenX: Int): Int? {
        if (filtered.isEmpty()) return null
        val chartW = (width - pad.left - pad.right).coerceAtLeast(1)
        val range = (viewEnd - viewStart).coerceAtLeast(1)
        val frac = (screenX - pad.left).toDouble() / chartW
        if (frac < 0.0 || frac > 1.0) return null
        return (viewStart + frac * range).toInt().coerceIn(viewStart, viewEnd)
    }

    private fun idxToScreenX(idx: Int, chartW: Int): Int? {
        if (idx !in viewStart..viewEnd) return null
        val range = (viewEnd - viewStart).coerceAtLeast(1)
        return pad.left + ((idx - viewStart).toDouble() / range * chartW).toInt()
    }

    private fun updateTooltip(point: Point) {
        val idx = screenXToIdx(point.x)
        if (idx == null || filtered.isEmpty()) {
            tooltipText = null
            return
        }
        tooltipText = tooltipLines(filtered[idx]).joinToString("\n")
        tooltipPos = point
    }

    override fun paintComponent(g: Graphics) {
        super.paintComponent(g)
        if (filtered.isEmpty()) {
            drawEmptyState(g as Graphics2D)
            return
        }

        val g2 = g as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
        g2.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE)

        val chartW = (width - pad.left - pad.right).coerceAtLeast(1)
        val chartH = (height - pad.top - pad.bottom).coerceAtLeast(1)
        val visible = filtered.subList(viewStart, viewEnd + 1)
        if (visible.isEmpty()) return

        var yLo = Double.MAX_VALUE
        var yHi = -Double.MAX_VALUE
        visible.forEach { point ->
            seriesDefs.filter { it.isVisible() }.forEach { def ->
                val v = def.valueOf(point)
                yLo = minOf(yLo, v)
                yHi = maxOf(yHi, v)
            }
        }
        val yRange = (yHi - yLo).let { if (it < 0.001) 1.0 else it }
        val yPad = yRange * 0.07
        yLo -= yPad
        yHi += yPad

        fun xOf(i: Int) = pad.left + i.toDouble() / (visible.size - 1).coerceAtLeast(1) * chartW
        fun yOf(v: Double) = pad.top + (1.0 - (v - yLo) / (yHi - yLo)) * chartH

        g2.color = Color(26, 26, 36)
        g2.fillRect(pad.left, pad.top, chartW, chartH)

        val yTicks = 7
        g2.font = Font("Monospaced", Font.PLAIN, 10)
        for (i in 0..yTicks) {
            val v = yLo + (yHi - yLo) * i / yTicks
            val y = yOf(v).toInt()
            g2.color = Color(44, 44, 58)
            g2.stroke = BasicStroke(0.7f)
            g2.drawLine(pad.left, y, pad.left + chartW, y)
            val label = "%.2f".format(v)
            g2.color = Color(150, 150, 178)
            g2.drawString(label, pad.left - g2.fontMetrics.stringWidth(label) - 7, y + 4)
        }

        val xTicks = minOf(8, visible.lastIndex.coerceAtLeast(1))
        for (i in 0..xTicks) {
            val idx = (i.toDouble() / xTicks * visible.lastIndex.coerceAtLeast(1)).toInt().coerceIn(0, visible.lastIndex)
            val x = xOf(idx).toInt()
            g2.color = Color(44, 44, 58)
            g2.stroke = BasicStroke(0.7f)
            g2.drawLine(x, pad.top, x, pad.top + chartH)
            val label = visible[idx].ts.format(axisFmt)
            g2.color = Color(150, 150, 178)
            g2.drawString(label, x - g2.fontMetrics.stringWidth(label) / 2, pad.top + chartH + 18)
        }

        if (anchorEnabled) {
            selectedIdx?.let { idx ->
                idxToScreenX(idx, chartW)?.let { ax ->
                    g2.color = Color(255, 210, 60, 30)
                    g2.fillRect(ax - 1, pad.top, 3, chartH)
                    g2.color = anchorColor
                    g2.stroke = BasicStroke(1.5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 0f, floatArrayOf(5f, 3f), 0f)
                    g2.drawLine(ax, pad.top, ax, pad.top + chartH)
                    val d = 6
                    val diamond = Polygon(
                        intArrayOf(ax, ax + d, ax, ax - d),
                        intArrayOf(pad.top - d, pad.top, pad.top + d, pad.top),
                        4
                    )
                    g2.fillPolygon(diamond)

                    g2.font = Font("Monospaced", Font.BOLD, 10)
                    val label = filtered[idx].ts.format(anchorFmt)
                    val w = g2.fontMetrics.stringWidth(label)
                    var lx = ax - w / 2
                    lx = lx.coerceIn(pad.left, pad.left + chartW - w)
                    g2.color = Color(40, 38, 22, 210)
                    g2.fillRoundRect(lx - 3, pad.top - d * 2 - 14, w + 6, 14, 4, 4)
                    g2.color = anchorColor
                    g2.drawString(label, lx, pad.top - d * 2 - 3)
                }
            }
        }

        val savedClip = g2.clip
        g2.clip = Rectangle(pad.left, pad.top, chartW, chartH)
        seriesDefs.filter { it.isVisible() }.forEach { def ->
            g2.color = def.color
            g2.stroke = if (def.dashed) {
                BasicStroke(def.strokeWidth, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND, 0f, floatArrayOf(8f, 4f), 0f)
            } else {
                BasicStroke(def.strokeWidth, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND)
            }
            drawSeries(g2, visible.map(def.valueOf), ::xOf, ::yOf)
        }
        g2.clip = savedClip

        g2.color = Color(85, 85, 108)
        g2.stroke = BasicStroke(1.5f)
        g2.drawRect(pad.left, pad.top, chartW, chartH)

        val savedTransform = g2.transform
        g2.color = Color(130, 130, 158)
        g2.font = Font("SansSerif", Font.PLAIN, 11)
        g2.rotate(-Math.PI / 2, 14.0, height / 2.0)
        g2.drawString(yAxisTitle, 14, height / 2)
        g2.transform = savedTransform

        drawLegend(g2)
        val tt = tooltipText
        val tp = tooltipPos
        if (tt != null && tp != null) drawTooltip(g2, tt, tp)
    }

    private fun drawSeries(g2: Graphics2D, values: List<Double>, xOf: (Int) -> Double, yOf: (Double) -> Double) {
        val path = GeneralPath()
        values.forEachIndexed { i, v ->
            val x = xOf(i).toFloat()
            val y = yOf(v).toFloat()
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        g2.draw(path)
    }

    private fun drawLegend(g2: Graphics2D) {
        g2.font = Font("SansSerif", Font.BOLD, 11)
        val fm = g2.fontMetrics
        var x = pad.left + 10
        val y = pad.top - 14
        seriesDefs.filter { it.isVisible() }.forEach { def ->
            g2.color = def.color
            g2.stroke = BasicStroke(2.5f)
            g2.drawLine(x, y - 4, x + 22, y - 4)
            g2.color = Color(210, 210, 232)
            g2.drawString(def.label, x + 26, y)
            x += 26 + fm.stringWidth(def.label) + 22
        }
    }

    private fun drawTooltip(g2: Graphics2D, text: String, pos: Point) {
        val lines = text.split("\n")
        g2.font = Font("Monospaced", Font.PLAIN, 11)
        val fm = g2.fontMetrics
        val w = lines.maxOf { fm.stringWidth(it) } + 20
        val h = lines.size * fm.height + 14
        var tx = pos.x + 14
        var ty = pos.y - h - 10
        if (tx + w > width - 6) tx = pos.x - w - 14
        if (ty < 6) ty = pos.y + 14

        g2.color = Color(30, 30, 44, 228)
        g2.fillRoundRect(tx, ty, w, h, 10, 10)
        g2.color = Color(75, 75, 100)
        g2.stroke = BasicStroke(1f)
        g2.drawRoundRect(tx, ty, w, h, 10, 10)
        g2.color = Color(200, 200, 222)
        lines.forEachIndexed { i, line ->
            g2.drawString(line, tx + 10, ty + fm.height * (i + 1) + 3)
        }
    }

    private fun drawEmptyState(g2: Graphics2D) {
        g2.color = Color(22, 22, 30)
        g2.fillRect(0, 0, width, height)
        g2.color = Color(120, 120, 145)
        g2.font = Font("SansSerif", Font.PLAIN, 14)
        val msg = "No data in the selected shared date range"
        val w = g2.fontMetrics.stringWidth(msg)
        g2.drawString(msg, (width - w) / 2, height / 2)
    }
}

// ------------------------------------------------------------
// Metric panels
// ------------------------------------------------------------

class PointMetricsPanel(private val methods: List<MethodConfig>) : JPanel() {
    private val labels = mutableMapOf<String, Triple<JLabel, JLabel, JLabel>>()

    init {
        layout = GridLayout(1, methods.size, 12, 0)
        background = Color(22, 22, 30)
        border = EmptyBorder(6, 8, 6, 8)

        methods.forEach { method ->
            val card = JPanel(GridLayout(4, 1, 3, 4)).apply {
                background = Color(30, 30, 44)
                border = BorderFactory.createCompoundBorder(
                    BorderFactory.createLineBorder(method.color.darker(), 1),
                    EmptyBorder(8, 14, 8, 14)
                )
            }
            val title = JLabel(method.label, SwingConstants.CENTER).apply {
                foreground = method.color
                font = Font("SansSerif", Font.BOLD, 13)
            }
            val mae = metricLabel(Color(155, 220, 155))
            val rmse = metricLabel(Color(155, 200, 230))
            val mape = metricLabel(Color(230, 200, 155))
            labels[method.key] = Triple(mae, rmse, mape)
            card.add(title)
            card.add(mae)
            card.add(rmse)
            card.add(mape)
            add(card)
        }
    }

    private fun metricLabel(color: Color) = JLabel("-").apply {
        foreground = color
        font = Font("Monospaced", Font.PLAIN, 12)
    }

    fun update(data: List<DataPoint>) {
        methods.forEach { method ->
            val (maeL, rmseL, mapeL) = labels[method.key] ?: return@forEach
            val metrics = computeMetrics(data.map { it.yTrue }, data.map { it.predictions[method.key] ?: 0.0 })
            maeL.text = "MAE:  %.4f".format(metrics.mae)
            rmseL.text = "RMSE: %.4f".format(metrics.rmse)
            mapeL.text = "MAPE: %.2f %%".format(metrics.mape)
        }
    }
}

class BessMetricsPanel : JPanel() {
    private val pred = metricChip(Color(166, 227, 161))
    private val truth = metricChip(Color(137, 180, 250))
    private val lost = metricChip(Color(243, 139, 168))
    private val avg = metricChip(Color(249, 226, 175))
    private val charge = metricChip(Color(203, 166, 247))
    private val discharge = metricChip(Color(180, 160, 240))
    private val days = metricChip(Color(210, 210, 232))

    init {
        layout = FlowLayout(FlowLayout.LEFT, 8, 6)
        background = Color(30, 30, 44)
        border = EmptyBorder(4, 8, 4, 8)
        add(label("Pred", pred))
        add(label("True", truth))
        add(label("Lost", lost))
        add(label("Avg/day", avg))
        add(vSep())
        add(label("Charge match", charge))
        add(label("Discharge match", discharge))
        add(vSep())
        add(label("Days", days))
    }

    private fun metricChip(color: Color) = JLabel("-").apply {
        foreground = color
        font = Font("Monospaced", Font.BOLD, 11)
    }

    private fun label(title: String, value: JLabel) = JPanel(FlowLayout(FlowLayout.LEFT, 4, 0)).apply {
        isOpaque = false
        add(JLabel("$title:").apply {
            foreground = Color(135, 135, 160)
            font = Font("SansSerif", Font.PLAIN, 10)
        })
        add(value)
    }

    fun update(summary: BessSummary) {
        pred.text = "%.2f €".format(summary.predProfit)
        truth.text = "%.2f €".format(summary.trueProfit)
        lost.text = "%.2f €".format(summary.lostProfit)
        avg.text = "%.2f €".format(summary.lostProfitAvg)
        charge.text = "%.1f %%".format(summary.chargeMatch * 100.0)
        discharge.text = "%.1f %%".format(summary.dischargeMatch * 100.0)
        days.text = summary.daysEvaluated.toString()
    }
}

// ------------------------------------------------------------
// UI helpers
// ------------------------------------------------------------

private fun darkButton(text: String) = JButton(text).apply {
    background = Color(46, 46, 66)
    foreground = Color(185, 185, 215)
    font = Font("SansSerif", Font.PLAIN, 11)
    isFocusPainted = false
    border = BorderFactory.createCompoundBorder(
        BorderFactory.createLineBorder(Color(75, 75, 100), 1),
        EmptyBorder(3, 10, 3, 10)
    )
}

private fun vSep() = JSeparator(SwingConstants.VERTICAL).apply {
    preferredSize = Dimension(1, 24)
    foreground = Color(65, 65, 88)
}

private fun dateSpinner(initial: LocalDate): JSpinner {
    val date = Date.from(initial.atStartOfDay(ZoneId.systemDefault()).toInstant())
    val model = SpinnerDateModel(date, null, null, java.util.Calendar.DAY_OF_MONTH)
    return JSpinner(model).apply {
        editor = JSpinner.DateEditor(this, "yyyy-MM-dd")
        preferredSize = Dimension(110, preferredSize.height)
    }
}

private fun JSpinner.localDateValue(): LocalDate =
    (value as Date).toInstant().atZone(ZoneId.systemDefault()).toLocalDate()

private fun JSpinner.setLocalDateValue(date: LocalDate) {
    value = Date.from(date.atStartOfDay(ZoneId.systemDefault()).toInstant())
}

// ------------------------------------------------------------
// Main application
// ------------------------------------------------------------

fun main() {
    val pointData = loadPointCsv("data/point_prediction.csv")
    if (pointData.isEmpty()) {
        System.err.println("ERROR: no point data loaded - expected data/point_prediction.csv")
        return
    }

    val bessSchedule = loadBessScheduleCsv("data/bess_schedule.csv")
    val dailyProfits = loadDailyProfitCsv("data/bess_daily_profit.csv")

    val methods = listOf(
        MethodConfig("hgbr", "HGBR", Color(80, 160, 255)),
        MethodConfig("catboost", "CatBoost", Color(255, 110, 90)),
        MethodConfig("xgboost", "XGBoost", Color(80, 215, 130))
    )

    val globalMin = listOf(pointData.first().ts.toLocalDate())
        .plus(bessSchedule.firstOrNull()?.ts?.toLocalDate())
        .plus(dailyProfits.firstOrNull()?.date)
        .filterNotNull()
        .minOrNull() ?: pointData.first().ts.toLocalDate()

    val globalMax = listOf(pointData.last().ts.toLocalDate())
        .plus(bessSchedule.lastOrNull()?.ts?.toLocalDate())
        .plus(dailyProfits.lastOrNull()?.date)
        .filterNotNull()
        .maxOrNull() ?: pointData.last().ts.toLocalDate()

    val forcedStart = LocalDate.parse("2025-07-01").coerceIn(globalMin, globalMax)
    val rangeModel = SharedRangeModel(forcedStart, globalMax)

    SwingUtilities.invokeLater {
        val frame = JFrame("Electricity Dashboard - Unified Kotlin View")
        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        frame.preferredSize = Dimension(1450, 980)

        val root = JPanel(BorderLayout(0, 10)).apply {
            background = Color(22, 22, 30)
            border = EmptyBorder(10, 12, 10, 12)
        }

        val pointChart = MultiSeriesTimeChart(
            source = pointData,
            seriesDefs = buildList {
                add(SeriesDef<DataPoint>("Actual", Color(230, 230, 250), { it.yTrue }, { true }, dashed = false, strokeWidth = 1.8f))
                methods.forEach { method ->
                    add(SeriesDef(method.label, method.color, { point -> point.predictions[method.key] ?: 0.0 }, { method.visible }, dashed = true))
                }
            },
            yAxisTitle = "Price (€/MWh)",
            tooltipLines = { point ->
                buildList {
                    add(point.ts.toLocalDateTime().toString().replace('T', ' ').substring(0, 16))
                    add("Actual:    %.2f €/MWh".format(point.yTrue))
                    methods.filter { it.visible }.forEach { method ->
                        add("%-9s: %.2f €/MWh".format(method.label, point.predictions[method.key] ?: 0.0))
                    }
                }
            },
            anchorEnabled = true
        )

        val pointMetrics = PointMetricsPanel(methods)

        val bessToggleState = linkedMapOf(
            "netPred" to true,
            "netTrue" to true,
            "chPred" to false,
            "disPred" to false,
            "chTrue" to false,
            "disTrue" to false,
        )

        val bessChart = MultiSeriesTimeChart(
            source = bessSchedule,
            seriesDefs = listOf(
                SeriesDef("Net Pred", Color(137, 180, 250), { it.netPred }, { bessToggleState["netPred"] == true }, false, 1.7f),
                SeriesDef("Net True", Color(166, 227, 161), { it.netTrue }, { bessToggleState["netTrue"] == true }, false, 1.7f),
                SeriesDef("Charge Pred", Color(203, 166, 247), { it.chargePred }, { bessToggleState["chPred"] == true }, true, 1.1f),
                SeriesDef("Discharge Pred", Color(249, 226, 175), { it.dischargePred }, { bessToggleState["disPred"] == true }, true, 1.1f),
                SeriesDef("Charge True", Color(176, 160, 240), { it.chargeTrue }, { bessToggleState["chTrue"] == true }, true, 1.0f),
                SeriesDef("Discharge True", Color(144, 216, 144), { it.dischargeTrue }, { bessToggleState["disTrue"] == true }, true, 1.0f),
            ),
            yAxisTitle = "Power (-dis / +chg) [MW]",
            tooltipLines = { point ->
                listOf(
                    point.ts.toLocalDateTime().toString().replace('T', ' ').substring(0, 16),
                    "Net Pred:   %.4f MW".format(point.netPred),
                    "Net True:   %.4f MW".format(point.netTrue),
                    "Charge Pred %.4f MW".format(point.chargePred),
                    "Dis Pred:   %.4f MW".format(point.dischargePred),
                    "Charge True %.4f MW".format(point.chargeTrue),
                    "Dis True:   %.4f MW".format(point.dischargeTrue)
                )
            }
        )

        val initialProfitPoints = buildProfitPointsForRange(
            dailyProfits,
            rangeModel.start,
            rangeModel.end
        )

        val profitChart = MultiSeriesTimeChart(
            source = initialProfitPoints,
            seriesDefs = listOf(
                SeriesDef("Pred (sched)", Color(166, 227, 161), { it.cumPredProfit }, { true }, false, 1.7f),
                SeriesDef("True (optimal)", Color(137, 180, 250), { it.cumTrueProfit }, { true }, false, 1.7f),
                SeriesDef("Lost profit", Color(243, 139, 168), { it.lostProfit }, { true }, true, 1.2f),
            ),
            yAxisTitle = "Cumulative Profit (?)",
            tooltipLines = { point ->
                listOf(
                    point.date.toString(),
                    "Pred: %.2f ?".format(point.cumPredProfit),
                    "True: %.2f ?".format(point.cumTrueProfit),
                    "Lost: %.2f ?".format(point.lostProfit)
                )
            }
        )

        val profitChart = MultiSeriesTimeChart(
            source = dailyProfits,
            seriesDefs = listOf(
                SeriesDef("Pred (sched)", Color(166, 227, 161), { it.cumPredProfit }, { true }, false, 1.7f),
                SeriesDef("True (optimal)", Color(137, 180, 250), { it.cumTrueProfit }, { true }, false, 1.7f),
                SeriesDef("Lost profit", Color(243, 139, 168), { it.lostProfit }, { true }, true, 1.2f),
            ),
            yAxisTitle = "Cumulative Profit (€)",
            tooltipLines = { point ->
                listOf(
                    point.date.toString(),
                    "Pred: %.2f €".format(point.cumPredProfit),
                    "True: %.2f €".format(point.cumTrueProfit),
                    "Lost: %.2f €".format(point.lostProfit)
                )
            }
        )

        val bessMetrics = BessMetricsPanel()
        if (bessSchedule.isNotEmpty() && dailyProfits.isNotEmpty()) {
            bessMetrics.update(computeBessSummary(bessSchedule, dailyProfits))
        }

        val rangeLabel = JLabel().apply {
            foreground = Color(155, 155, 185)
            font = Font("Monospaced", Font.PLAIN, 11)
        }
        val fromSpinner = dateSpinner(rangeModel.start)
        val toSpinner = dateSpinner(rangeModel.end)
        val sharedRangeInfo = JLabel("Shared time range").apply {
            foreground = Color(100, 100, 128)
            font = Font("SansSerif", Font.BOLD, 11)
        }

        val anchorLabel = JLabel("click on the top chart to select an anchor").apply {
            foreground = Color(130, 125, 80)
            font = Font("Monospaced", Font.PLAIN, 11)
        }

        val quickButtons = listOf(
            "1 Day" to Duration.ofDays(1),
            "1 Week" to Duration.ofDays(7),
            "1 Month" to Duration.ofDays(30)
        ).map { (text, duration) ->
            darkButton(text).apply {
                isEnabled = false
                addActionListener { pointChart.zoomFromSelected(duration) }
            }
        }

        val applySharedRange: () -> Unit = {
            val start = fromSpinner.localDateValue().coerceIn(globalMin, globalMax)
            val end = toSpinner.localDateValue().coerceIn(globalMin, globalMax)
            rangeModel.update(start, end)
        }

        fun syncSharedRangeUi(start: LocalDate, end: LocalDate) {
            fromSpinner.setLocalDateValue(start)
            toSpinner.setLocalDateValue(end)
            rangeLabel.text = "Shared range | $start → $end"
        }

        rangeModel.onChange { start, end ->
            syncSharedRangeUi(start, end)
            pointChart.setDateFilter(start, end)
            pointChart.resetZoom()

            pointMetrics.update(pointChart.filtered)

            bessChart.setDateFilter(start, end)
            bessChart.resetZoom()

            val profitPoints = buildProfitPointsForRange(dailyProfits, start, end)
            profitChart.replaceSource(profitPoints)
            profitChart.resetZoom()

            if (bessChart.filtered.isNotEmpty() && profitChart.filtered.isNotEmpty()) {
                val dailySlice = profitChart.filtered.filterIsInstance<DailyProfitPoint>()
                val bessSlice = bessChart.filtered.filterIsInstance<BessSchedulePoint>()
                bessMetrics.update(computeBessSummary(bessSlice, dailySlice))
            }
        }

        val topToolbar = JPanel().apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            background = Color(30, 30, 44)
            border = BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(Color(52, 52, 72), 1),
                EmptyBorder(2, 4, 2, 4)
            )
        }

        val row1 = JPanel(FlowLayout(FlowLayout.LEFT, 14, 5)).apply {
            background = Color(30, 30, 44)
            add(sharedRangeInfo)
            add(rangeLabel)
            add(vSep())
            add(JLabel("From").apply { foreground = Color(155, 155, 185) })
            add(fromSpinner)
            add(JLabel("To").apply { foreground = Color(155, 155, 185) })
            add(toSpinner)

            val applyBtn = darkButton("Apply Range")
            applyBtn.addActionListener { applySharedRange() }
            add(applyBtn)

            val nextDayBtn = darkButton("+1 Day")
            nextDayBtn.addActionListener { rangeModel.shiftDays(1, globalMin, globalMax) }
            add(nextDayBtn)

            val resetRangeBtn = darkButton("Reset Range")
            resetRangeBtn.addActionListener { rangeModel.update(forcedStart, globalMax) }
            add(resetRangeBtn)

            val resetZoomBtn = darkButton("Reset Zoom")
            resetZoomBtn.addActionListener {
                pointChart.resetZoom()
                bessChart.resetZoom()
                profitChart.resetZoom()
            }
            add(resetZoomBtn)
        }

        val row2 = JPanel(FlowLayout(FlowLayout.LEFT, 14, 5)).apply {
            background = Color(30, 30, 44)
            add(JLabel("Point forcast").apply {
                foreground = Color(155, 155, 185)
                font = Font("SansSerif", Font.BOLD, 11)
            })
            methods.forEach { method ->
                val cb = JCheckBox(method.label, true).apply {
                    foreground = method.color
                    background = Color(30, 30, 44)
                    font = Font("SansSerif", Font.BOLD, 12)
                    isFocusPainted = false
                    addActionListener {
                        method.visible = isSelected
                        pointChart.repaint()
                    }
                }
                add(cb)
            }
            add(vSep())
            add(JLabel("Anchor").apply {
                foreground = Color(255, 210, 60)
                font = Font("SansSerif", Font.BOLD, 11)
            })
            add(anchorLabel)
            quickButtons.forEach { add(it) }
        }

        val row3 = JPanel(FlowLayout(FlowLayout.LEFT, 14, 5)).apply {
            background = Color(30, 30, 44)
            add(JLabel("BESS series").apply {
                foreground = Color(155, 155, 185)
                font = Font("SansSerif", Font.BOLD, 11)
            })
            listOf(
                "netPred" to Pair("Net Pred", Color(137, 180, 250)),
                "netTrue" to Pair("Net True", Color(166, 227, 161)),
                "chPred" to Pair("Chg Pred", Color(203, 166, 247)),
                "disPred" to Pair("Dis Pred", Color(249, 226, 175)),
                "chTrue" to Pair("Chg True", Color(176, 160, 240)),
                "disTrue" to Pair("Dis True", Color(144, 216, 144))
            ).forEach { (key, meta) ->
                val cb = JCheckBox(meta.first, bessToggleState[key] == true).apply {
                    foreground = meta.second
                    background = Color(30, 30, 44)
                    font = Font("SansSerif", Font.BOLD, 12)
                    isFocusPainted = false
                    addActionListener {
                        bessToggleState[key] = isSelected
                        bessChart.repaint()
                    }
                }
                add(cb)
            }
        }

        topToolbar.add(row1)
        topToolbar.add(JSeparator().apply { foreground = Color(45, 45, 62) })
        topToolbar.add(row2)
        topToolbar.add(JSeparator().apply { foreground = Color(45, 45, 62) })
        topToolbar.add(row3)

        pointChart.onViewChanged = { visible ->
            pointMetrics.update(visible.filterIsInstance<DataPoint>())
        }
        pointChart.onPointSelected = { point ->
            anchorLabel.text = point.ts.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"))
            anchorLabel.foreground = Color(255, 210, 60)
            quickButtons.forEach { it.isEnabled = true }
        }

        val pointSection = JPanel(BorderLayout(0, 4)).apply {
            background = Color(22, 22, 30)
            add(JLabel("  Price Prediction").apply {
                foreground = Color(100, 100, 128)
                font = Font("SansSerif", Font.BOLD, 12)
            }, BorderLayout.NORTH)
            add(pointChart, BorderLayout.CENTER)
        }

        val pointMetricsWrap = JPanel(BorderLayout()).apply {
            background = Color(22, 22, 30)
            add(JLabel("  Performance metrics (visible window)").apply {
                foreground = Color(100, 100, 128)
                font = Font("SansSerif", Font.PLAIN, 11)
            }, BorderLayout.NORTH)
            add(pointMetrics, BorderLayout.CENTER)
            preferredSize = Dimension(0, 115)
        }

        val bottomCharts = JPanel(GridLayout(1, 2, 10, 0)).apply {
            background = Color(22, 22, 30)
            add(JPanel(BorderLayout()).apply {
                background = Color(22, 22, 30)
                add(JLabel("  BESS Schedule").apply {
                    foreground = Color(100, 100, 128)
                    font = Font("SansSerif", Font.BOLD, 12)
                }, BorderLayout.NORTH)
                add(bessChart, BorderLayout.CENTER)
            })
            add(JPanel(BorderLayout()).apply {
                background = Color(22, 22, 30)
                add(JLabel("  Cumulative Profit").apply {
                    foreground = Color(100, 100, 128)
                    font = Font("SansSerif", Font.BOLD, 12)
                }, BorderLayout.NORTH)
                add(profitChart, BorderLayout.CENTER)
            })
        }

        val center = JPanel(GridLayout(2, 1, 0, 10)).apply {
            background = Color(22, 22, 30)
            add(pointSection)
            add(bottomCharts)
        }

        root.add(topToolbar, BorderLayout.NORTH)
        root.add(center, BorderLayout.CENTER)
        root.add(JPanel(BorderLayout()).apply {
            background = Color(22, 22, 30)
            add(pointMetricsWrap, BorderLayout.NORTH)
            add(bessMetrics, BorderLayout.SOUTH)
        }, BorderLayout.SOUTH)

        frame.contentPane = root
        syncSharedRangeUi(rangeModel.start, rangeModel.end)
        pointChart.setDateFilter(rangeModel.start, rangeModel.end)
        pointChart.resetZoom()
        pointMetrics.update(pointChart.filtered)
        bessChart.setDateFilter(rangeModel.start, rangeModel.end)
        bessChart.resetZoom()
        profitChart.replaceSource(
            buildProfitPointsForRange(dailyProfits, rangeModel.start, rangeModel.end)
        )
        profitChart.resetZoom()
        if (bessChart.filtered.isNotEmpty() && profitChart.filtered.isNotEmpty()) {
            bessMetrics.update(
                computeBessSummary(
                    bessChart.filtered.filterIsInstance<BessSchedulePoint>(),
                    profitChart.filtered.filterIsInstance<DailyProfitPoint>()
                )
            )
        }
        frame.pack()
        frame.setLocationRelativeTo(null)
        frame.isVisible = true
    }
}

private fun LocalDate.coerceIn(min: LocalDate, max: LocalDate): LocalDate = when {
    this < min -> min
    this > max -> max
    else -> this
}