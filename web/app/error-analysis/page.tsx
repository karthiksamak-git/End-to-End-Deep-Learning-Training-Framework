import { CalibrationChart } from "@/components/calibration-chart";
import { ConfusionMatrix } from "@/components/confusion-matrix";
import { PixelImage } from "@/components/pixel-image";
import {
  Badge,
  Card,
  DataTable,
  EmptyState,
  PageHeader,
  SectionTitle,
  Stat,
  formatCount,
  formatNumber,
  formatPct,
} from "@/components/ui";
import { getErrorAnalysis } from "@/lib/data";

export const dynamic = "force-static";

export const metadata = { title: "Error analysis" };

function plain(markdown: string): string {
  return markdown.replace(/\*\*/g, "").replace(/^- /, "");
}

export default async function ErrorAnalysisPage() {
  const analysis = await getErrorAnalysis();

  if (!analysis) {
    return (
      <>
        <PageHeader
          eyebrow="Error analysis"
          title="What the model gets wrong"
          description="Run `dltf analyze --config configs/image_cnn.yaml --task image` to generate this report."
        />
        <EmptyState
          title="No analysis yet"
          detail="The analyze command writes per-class metrics, a confusion matrix, misclassified samples and calibration data."
        />
      </>
    );
  }

  const findings = analysis.findings_markdown.split("\n").filter(Boolean);
  const gallery = analysis.misclassified.items.filter((item) => item.pixels);

  return (
    <>
      <PageHeader
        eyebrow="Error analysis"
        title="Where the network is wrong — and how confident it is"
        description={
          `Held-out ${analysis.task} test split (${formatCount(analysis.num_samples)} samples). ` +
          "The gallery shows the most confident mistakes, with the pixels exactly as the model saw them after normalisation was inverted."
        }
      />

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Test accuracy" value={formatPct(analysis.overall.accuracy)} accent />
        <Stat label="Macro F1" value={formatPct(analysis.overall.f1_macro)} />
        <Stat
          label="Misclassified"
          value={formatCount(analysis.num_misclassified)}
          sub={`of ${formatCount(analysis.num_samples)} samples`}
        />
        <Stat
          label="Calibration (ECE)"
          value={formatNumber(analysis.calibration.ece, 4)}
          sub={`${analysis.calibration.bins.length} confidence bins`}
        />
      </section>

      <section className="mt-8 grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <SectionTitle
            title="Confusion matrix"
            hint="Rows: true class · columns: predicted class · click a cell"
          />
          <ConfusionMatrix
            labels={analysis.confusion_matrix.labels}
            matrix={analysis.confusion_matrix.matrix}
          />
        </Card>
        <Card>
          <SectionTitle title="Automated findings" hint="Generated from these numbers" />
          <ul className="space-y-2 text-sm">
            {findings.map((line) => (
              <li key={line} className="flex gap-2">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-500" />
                <span className="leading-relaxed">{plain(line)}</span>
              </li>
            ))}
          </ul>
          {analysis.top_confusions.length > 0 ? (
            <div className="mt-4 space-y-1">
              <p className="text-xs font-semibold uppercase tracking-wide text-ink-700/60 dark:text-ink-100/50">
                Most confused pairs
              </p>
              {analysis.top_confusions.slice(0, 4).map((pair) => (
                <div key={`${pair.true_label}-${pair.pred_label}`} className="flex justify-between text-sm">
                  <span>
                    {pair.true_label} → {pair.pred_label}
                  </span>
                  <span className="tabular-nums">{pair.count}</span>
                </div>
              ))}
            </div>
          ) : null}
        </Card>
      </section>

      <section className="mt-6">
        <SectionTitle
          title="Per-class metrics"
          hint="Precision, recall, F1 and support from the held-out split"
        />
        <Card>
          <DataTable
            head={["Class", "Precision", "Recall", "F1", "Support", "Recall bar"]}
            rows={analysis.per_class.map((row) => [
              row.class,
              formatNumber(row.precision, 4),
              formatNumber(row.recall, 4),
              formatNumber(row.f1, 4),
              formatCount(row.support),
              <div
                key={`${row.class}-bar`}
                className="h-2 w-32 overflow-hidden rounded-full bg-ink-100 dark:bg-white/10"
              >
                <div
                  className="h-full rounded-full bg-emerald-500"
                  style={{ width: `${Math.max(2, row.recall * 100)}%` }}
                />
              </div>,
            ])}
          />
        </Card>
      </section>

      <section className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <SectionTitle
            title="Most confident mistakes"
            hint={`${gallery.length} samples · true label in red, prediction in grey`}
          />
          {gallery.length === 0 ? (
            <p className="muted">No pixel samples were exported for this analysis.</p>
          ) : (
            <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 lg:grid-cols-5">
              {gallery.map((item, index) => (
                <figure key={`${item.sample_index}-${index}`} className="text-center">
                  <div className="overflow-hidden rounded-lg border border-ink-200/70 bg-black p-1 dark:border-white/10">
                    <PixelImage
                      pixels={item.pixels ?? []}
                      size={64}
                      label={`True ${item.true_label}, predicted ${item.pred_label}`}
                    />
                  </div>
                  <figcaption className="mt-1 text-[10px] leading-tight">
                    <span className="block font-medium text-rose-600 dark:text-rose-400">
                      {item.true_label}
                    </span>
                    <span className="block text-ink-700/60 dark:text-ink-100/50">
                      {item.pred_label} · {(item.confidence * 100).toFixed(0)}%
                    </span>
                  </figcaption>
                </figure>
              ))}
            </div>
          )}
        </Card>

        <div className="space-y-6">
          <Card>
            <SectionTitle
              title="Reliability diagram"
              hint={`ECE ${formatNumber(analysis.calibration.ece, 4)} · dashed line is perfect calibration`}
            />
            <CalibrationChart bins={analysis.calibration.bins} />
          </Card>
          <Card>
            <SectionTitle title="Confidence detail" hint="Top mistakes with their top-3 predictions" />
            <div className="space-y-3">
              {analysis.misclassified.items.slice(0, 3).map((item, index) => (
                <div key={`${item.sample_index}-detail-${index}`} className="text-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone="warn">{item.true_label}</Badge>
                    <span className="muted">predicted</span>
                    <Badge tone="info">{item.pred_label}</Badge>
                    <span className="muted tabular-nums">{(item.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="mt-1.5 space-y-1">
                    {item.top3.map((top) => (
                      <div key={top.label} className="flex items-center gap-2">
                        <span className="w-28 shrink-0 truncate text-xs">{top.label}</span>
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-100 dark:bg-white/10">
                          <div
                            className="h-full rounded-full bg-indigo-500"
                            style={{ width: `${Math.max(2, top.probability * 100)}%` }}
                          />
                        </div>
                        <span className="w-12 text-right text-xs tabular-nums">
                          {(top.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </section>
    </>
  );
}
