import { ExperimentExplorer } from "@/components/experiment-explorer";
import {
  Card,
  DataTable,
  EmptyState,
  PageHeader,
  SectionTitle,
  formatCount,
  formatNumber,
  formatPct,
  formatSeconds,
} from "@/components/ui";
import { getRuns } from "@/lib/data";

export const dynamic = "force-static";

export const metadata = { title: "Experiments" };

export default async function ExperimentsPage() {
  const runs = await getRuns();

  return (
    <>
      <PageHeader
        eyebrow="Experiments"
        title="Training and validation curves"
        description="Per-epoch metrics are written to metrics.csv and metrics.json by the trainer, then summarised here. Every run stores its full config, git commit, seed and device so it can be reproduced."
      />

      {runs.length === 0 ? (
        <EmptyState
          title="No runs found"
          detail="Run `dltf train --config configs/image_cnn.yaml` and re-export the artifacts."
        />
      ) : (
        <>
          <ExperimentExplorer runs={runs} />

          <section className="mt-10">
            <SectionTitle
              title="Run comparison"
              hint="Held-out test metrics, not validation: the test split is never used for model selection"
            />
            <Card>
              <DataTable
                head={[
                  "Run",
                  "Task",
                  "Epochs",
                  "Best epoch",
                  "Test acc.",
                  "Test F1",
                  "ROC-AUC",
                  "Train size",
                  "Duration",
                ]}
                rows={runs.map((run) => [
                  <span key={`${run.id}-name`} className="font-medium">
                    {run.name}
                  </span>,
                  run.task ?? "—",
                  String(run.epochs_run ?? "—"),
                  String(run.best.epoch ?? "—"),
                  formatPct(run.test_metrics.accuracy),
                  formatPct(run.test_metrics.f1_macro),
                  formatNumber(run.test_metrics.roc_auc ?? null, 4),
                  formatCount(run.sizes.train),
                  formatSeconds(run.duration_sec),
                ])}
              />
            </Card>
          </section>

          <section className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {runs.map((run) => (
              <Card key={run.id} className="p-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-ink-700/60 dark:text-ink-100/50">
                  {run.name}
                </p>
                <p className="muted mt-1 font-mono text-[11px]">{run.id}</p>
                <dl className="mt-3 space-y-1 text-sm">
                  <div className="flex justify-between gap-2">
                    <dt className="muted">Seed</dt>
                    <dd className="tabular-nums">{run.seed ?? "—"}</dd>
                  </div>
                  <div className="flex justify-between gap-2">
                    <dt className="muted">Device</dt>
                    <dd className="tabular-nums">{run.device.device ?? "cpu"}</dd>
                  </div>
                  <div className="flex justify-between gap-2">
                    <dt className="muted">Params</dt>
                    <dd className="tabular-nums">{formatCount(run.model.parameters_total)}</dd>
                  </div>
                  <div className="flex justify-between gap-2">
                    <dt className="muted">Commit</dt>
                    <dd className="font-mono text-xs">{run.git_hash?.slice(0, 7) ?? "—"}</dd>
                  </div>
                </dl>
              </Card>
            ))}
          </section>
        </>
      )}
    </>
  );
}
