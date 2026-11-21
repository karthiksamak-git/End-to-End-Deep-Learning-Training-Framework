import { readFile } from "node:fs/promises";
import path from "node:path";

import type {
  CvFile,
  ErrorAnalysisFile,
  ExperimentsFile,
  Headline,
  HpoFile,
  ModelCard,
  PlaygroundSample,
} from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

async function readJson<T>(name: string): Promise<T | null> {
  try {
    const raw = await readFile(path.join(DATA_DIR, name), "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export async function getHeadline(): Promise<Headline | null> {
  return readJson<Headline>("headline.json");
}

export async function getRuns(): Promise<ExperimentsFile["runs"]> {
  const file = await readJson<ExperimentsFile>("experiments.json");
  return file?.runs ?? [];
}

export async function getCv(): Promise<CvFile | null> {
  return readJson<CvFile>("cv_results.json");
}

export async function getHpo(): Promise<HpoFile | null> {
  return readJson<HpoFile>("hpo_trials.json");
}

export async function getErrorAnalysis(): Promise<ErrorAnalysisFile | null> {
  return readJson<ErrorAnalysisFile>("error_analysis.json");
}

export async function getModelCard(): Promise<ModelCard | null> {
  return readJson<ModelCard>("model_card.json");
}

export async function getPlaygroundSamples(): Promise<PlaygroundSample[]> {
  const file = await readJson<{ samples: PlaygroundSample[] }>("playground_samples.json");
  return file?.samples ?? [];
}
