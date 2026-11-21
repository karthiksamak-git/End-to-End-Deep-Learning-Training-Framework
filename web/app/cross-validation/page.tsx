import { CvChart } from "@/components/cv-chart";
import {
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
import { getCv } from "@/lib/data";

export const dynamic = "force-static";

export const metadata = { title: "Cross-validation" };

export default async function CrossValidationPage() {
  const cv = await getCv();

  if (!cv) {
    return (
      <>
        <PageHeader
          eyebrow="Cross-validation"
          title="Stratified k-fold cross-validation"
          description="Run `dltf cv --config configs/tabular_mlp.yaml` to generate these results."
        />
        <EmptyState
          title="No CV results yet"
          detail="The CV command writes artifacts/cv_results.json and copies it into web/public/data."
        />
      </>
    );
  }

  const summary = cv.summary;
  const metric = (name: string) => summary[name] ?? { mean: 0, std: 0 };

  return (
    <>
      <PageHeader
        eyebrow="Cross-validation"
        title={`${cv.n_splits}-fold stratified CV on ${cv.dataset}`}
        description="Preprocessing is refitted inside every fold and each fold's evaluation rows are never seen during training — including for early stopping, which uses an inner validation split of the fold's training portion. Reported numbers are therefore honest out-of-fold estimates."
      />

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="Accuracy"
          value={formatPct(metric("accuracy").mean)}
          sub={`± ${(metric("accuracy").std * 100).toFixed(2)} pts`}
          accent
        />
        <Stat
          label="Macro F1"
          value={formatPct(metric("f1_macro").mean)}
          sub={`± ${(metric("f1_macro").std * 100).toFixed(2)} pts`}
        />
        <Stat
          label="ROC-AUC"
          value={formatNumber(metric("roc_auc").mean, 4)}
          sub={`± ${formatNumber(metric("roc_auc").std, 4)}`}
        />
        <Stat
          label="Log loss"
          value={formatNumber(metric("log_loss").mean, 4)}
          sub={`± ${formatNumber(metric("log_loss").std, 4)}`}
        />
      </section>

      <section className="mt-8">
        <SectionTitle
          title="Per-fold metrics"
          hint={`${cv.epochs} epochs per fold · model ${cv.model} · seed ${cv.seed}`}
        />
        <Card>
          <CvChart folds={cv.folds} />
          <p className="muted mt-2 text-center">y-axis starts at 0.80 for readability</p>
        </Card>
      </section>

      <section className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <SectionTitle title="Fold details" />
          <DataTable
            head={["Fold", "Train", "Val", "Test", "Best epoch", "Accuracy", "F1", "AUC"]}
            rows={cv.folds.map((fold) => [
              String(fold.fold + 1),
              formatCount(fold.train_size),
              formatCount(fold.val_size),
              formatCount(fold.test_size),
              String(fold.best_epoch),
              formatPct(fold.accuracy),
              formatPct(fold.f1_macro),
              formatNumber(fold.roc_auc, 4),
            ])}
          />
        </Card>
        <Card>
          <SectionTitle title="Mean ± std" hint="Population std across folds" />
          <DataTable
            head={["Metric", "Mean", "Std"]}
            rows={Object.entries(summary).map(([name, stats]) => [
              name.replace(/_/g, " "),
              formatNumber(stats.mean, 4),
              formatNumber(stats.std, 4),
            ])}
          />
          <p className="muted mt-3">
            Generated {new Date(cv.generated_at).toLocaleString("en-GB")} · device{" "}
            {cv.device?.device ?? "cpu"}
          </p>
        </Card>
      </section>
    </>
  );
}
