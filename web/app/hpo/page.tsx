import { ImportanceChart, TrialScatter } from "@/components/hpo-charts";
import {
  Badge,
  Card,
  DataTable,
  EmptyState,
  PageHeader,
  SectionTitle,
  Stat,
  formatNumber,
} from "@/components/ui";
import { getHpo } from "@/lib/data";

export const dynamic = "force-static";

export const metadata = { title: "Hyper-parameter search" };

function formatParam(value: string | number): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toPrecision(4);
  }
  return String(value);
}

export default async function HpoPage() {
  const hpo = await getHpo();

  if (!hpo) {
    return (
      <>
        <PageHeader
          eyebrow="Hyper-parameter search"
          title="Optuna TPE with median pruning"
          description="Run `dltf hpo --config configs/hpo_image_cnn.yaml` to generate these results."
        />
        <EmptyState
          title="No study results yet"
          detail="The HPO command writes artifacts/hpo_results.json and copies it into web/public/data."
        />
      </>
    );
  }

  return (
    <>
      <PageHeader
        eyebrow="Hyper-parameter search"
        title={`${hpo.n_trials} trials on a CPU-sized Fashion-MNIST subset`}
        description={`Seeded TPE sampler, median pruner, ${hpo.epochs_per_trial} epochs per trial and ${hpo.subset_train} training images per trial — enough to compare configurations without burning the CPU budget. The objective is ${hpo.metric} on the validation split; the winning configuration is then used for the full image run.`}
      />

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label={`Best ${hpo.metric}`}
          value={formatNumber(hpo.best_value, 4)}
          sub={`trial ${hpo.best_trial ?? "—"}`}
          accent
        />
        <Stat
          label="Completed"
          value={`${hpo.n_completed} / ${hpo.n_trials}`}
          sub={`${hpo.n_pruned} pruned by the median pruner`}
        />
        <Stat
          label="Search space"
          value={`${Object.keys(hpo.search_space).length} params`}
          sub={Object.keys(hpo.search_space).join(", ")}
        />
        <Stat label="Sampler seed" value={String(hpo.seed)} sub="TPE, multivariate" />
      </section>

      <section className="mt-8 grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <SectionTitle
            title="Trial history"
            hint={`Objective per trial · direction: ${hpo.direction}`}
          />
          <TrialScatter hpo={hpo} />
        </Card>
        <Card>
          <SectionTitle title="Best parameters" hint="Fed into configs/image_cnn.yaml" />
          <DataTable
            head={["Parameter", "Value"]}
            minWidth={280}
            rows={Object.entries(hpo.best_params).map(([name, value]) => [
              name,
              <span key={name} className="font-mono text-xs">
                {formatParam(value)}
              </span>,
            ])}
          />
          <p className="muted mt-3">
            {hpo.n_pruned > 0
              ? `Pruning stopped ${hpo.n_pruned} trial(s) early once they fell behind the median.`
              : "No trials were pruned in this study."}
          </p>
        </Card>
      </section>

      <section className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <SectionTitle title="Parameter importance" hint="fANOVA on completed trials" />
          <ImportanceChart hpo={hpo} />
        </Card>
        <Card>
          <SectionTitle title="Search space" hint="Declared in configs/hpo_image_cnn.yaml" />
          <DataTable
            head={["Parameter", "Range / choices"]}
            rows={Object.entries(hpo.search_space).map(([name, values]) => [
              name,
              <span key={name} className="font-mono text-xs">
                {values.join(" → ")}
              </span>,
            ])}
          />
        </Card>
      </section>

      <section className="mt-6">
        <SectionTitle title="All trials" hint="Include pruned trials and their duration" />
        <Card>
          <DataTable
            head={["Trial", "State", "Objective", "Params", "Duration"]}
            rows={hpo.trials.map((trial) => [
              String(trial.number),
              <Badge
                key={`state-${trial.number}`}
                tone={trial.state === "complete" ? "good" : trial.state === "pruned" ? "warn" : "neutral"}
              >
                {trial.state}
              </Badge>,
              trial.value === null ? "—" : formatNumber(trial.value, 4),
              <span key={`params-${trial.number}`} className="font-mono text-[11px]">
                {Object.entries(trial.params)
                  .map(([key, value]) => `${key}=${formatParam(value)}`)
                  .join(" · ")}
              </span>,
              `${trial.duration_sec.toFixed(1)} s`,
            ])}
          />
        </Card>
      </section>
    </>
  );
}
