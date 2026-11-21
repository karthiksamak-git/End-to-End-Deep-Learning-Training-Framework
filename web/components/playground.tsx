"use client";

import type { InferenceSession } from "onnxruntime-web";
import { useCallback, useEffect, useRef, useState } from "react";

import { Badge } from "@/components/ui";
import type { PlaygroundSample } from "@/lib/types";

const CANVAS = 28;
const DISPLAY = 280;

type Prediction = { label: string; probability: number };
type Status = "idle" | "loading" | "ready" | "error";

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

export function Playground({
  samples,
  classNames,
  mean,
  std,
  modelSizeMb,
  parityDiff,
}: {
  samples: PlaygroundSample[];
  classNames: string[];
  mean: number;
  std: number;
  modelSizeMb: number;
  parityDiff: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sessionRef = useRef<InferenceSession | null>(null);
  const ortRef = useRef<typeof import("onnxruntime-web") | null>(null);
  const drawingRef = useRef(false);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);

  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction[] | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [source, setSource] = useState<string>("drawing");

  // Paint the initial black canvas once mounted.
  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!context) return;
    context.fillStyle = "#000000";
    context.fillRect(0, 0, CANVAS, CANVAS);
  }, []);

  const ensureSession = useCallback(async (): Promise<InferenceSession> => {
    if (sessionRef.current) return sessionRef.current;
    setStatus("loading");
    setError(null);
    try {
      const ort = await import("onnxruntime-web");
      ort.env.wasm.wasmPaths = "/ort/";
      ort.env.wasm.numThreads = 1;
      ort.env.logLevel = "error";
      const session = await ort.InferenceSession.create("/models/model.onnx", {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      ortRef.current = ort;
      sessionRef.current = session;
      setStatus("ready");
      return session;
    } catch (cause) {
      setStatus("error");
      const message = cause instanceof Error ? cause.message : String(cause);
      setError(`Could not load the ONNX model: ${message}`);
      throw cause;
    }
  }, []);

  // Warm the model up as soon as the page mounts, so the first click is instant.
  useEffect(() => {
    void ensureSession().catch(() => undefined);
  }, [ensureSession]);

  const clearCanvas = useCallback(() => {
    const context = canvasRef.current?.getContext("2d");
    if (!context) return;
    context.fillStyle = "#000000";
    context.fillRect(0, 0, CANVAS, CANVAS);
    setPrediction(null);
    setLatency(null);
    setSource("drawing");
  }, []);

  const loadPixels = useCallback((pixels: number[], label: string) => {
    const context = canvasRef.current?.getContext("2d");
    if (!context) return;
    const image = context.createImageData(CANVAS, CANVAS);
    for (let i = 0; i < CANVAS * CANVAS; i += 1) {
      const value = pixels[i] ?? 0;
      image.data[i * 4] = value;
      image.data[i * 4 + 1] = value;
      image.data[i * 4 + 2] = value;
      image.data[i * 4 + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    setSource(`sample: ${label}`);
    setPrediction(null);
  }, []);

  const modelInput = useCallback((): Float32Array | null => {
    const context = canvasRef.current?.getContext("2d");
    if (!context) return null;
    const { data } = context.getImageData(0, 0, CANVAS, CANVAS);
    const input = new Float32Array(CANVAS * CANVAS);
    for (let i = 0; i < CANVAS * CANVAS; i += 1) {
      // Fashion-MNIST is single-channel; the exporter used train-split mean/std.
      input[i] = (data[i * 4] / 255 - mean) / std;
    }
    return input;
  }, [mean, std]);

  const predict = useCallback(async () => {
    setPredicting(true);
    setError(null);
    try {
      const session = await ensureSession();
      const ort = ortRef.current;
      if (!ort) throw new Error("onnxruntime-web failed to initialise");
      const input = modelInput();
      if (!input) throw new Error("canvas is not ready");

      const tensor = new ort.Tensor("float32", input, [1, 1, CANVAS, CANVAS]);
      const started = performance.now();
      const outputs = await session.run({ [session.inputNames[0]]: tensor });
      const logits = Array.from(
        outputs[session.outputNames[0]].data as Float32Array,
      ) as number[];
      const probabilities = softmax(logits);
      setLatency(performance.now() - started);
      setPrediction(
        probabilities
          .map((probability, index) => ({
            label: classNames[index] ?? `class ${index}`,
            probability,
          }))
          .sort((a, b) => b.probability - a.probability)
          .slice(0, 3),
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setStatus("error");
    } finally {
      setPredicting(false);
    }
  }, [classNames, ensureSession, modelInput]);

  const pointFromEvent = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * CANVAS,
      y: ((event.clientY - rect.top) / rect.height) * CANVAS,
    };
  };

  const paint = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const context = canvasRef.current?.getContext("2d");
    if (!context) return;
    const point = pointFromEvent(event);
    const previous = lastPointRef.current ?? point;
    context.strokeStyle = "#ffffff";
    context.lineWidth = 2.4;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.beginPath();
    context.moveTo(previous.x, previous.y);
    context.lineTo(point.x, point.y);
    context.stroke();
    lastPointRef.current = point;
    setSource("drawing");
  };

  const top = prediction?.[0];

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <div className="lg:col-span-2">
        <div className="rounded-xl border border-ink-200/80 bg-white/85 p-5 shadow-sm dark:border-white/10 dark:bg-white/[0.04]">
          <div className="flex items-center justify-between gap-2">
            <h2 className="font-semibold">Draw a garment</h2>
            <Badge tone={status === "ready" ? "good" : status === "error" ? "warn" : "info"}>
              {status === "ready" ? "model ready" : status === "loading" ? "loading…" : status}
            </Badge>
          </div>
          <p className="muted mt-1">
            White on black, exactly like Fashion-MNIST. The canvas is 28×28 pixels — the native
            input size of the model.
          </p>
          <canvas
            ref={canvasRef}
            width={CANVAS}
            height={CANVAS}
            onPointerDown={(event) => {
              drawingRef.current = true;
              lastPointRef.current = pointFromEvent(event);
              event.currentTarget.setPointerCapture(event.pointerId);
              paint(event);
            }}
            onPointerMove={(event) => {
              if (drawingRef.current) paint(event);
            }}
            onPointerUp={() => {
              drawingRef.current = false;
              lastPointRef.current = null;
            }}
            onPointerLeave={() => {
              drawingRef.current = false;
              lastPointRef.current = null;
            }}
            className="mx-auto mt-4 block cursor-crosshair touch-none rounded-lg bg-black ring-1 ring-inset ring-white/10"
            style={{ width: DISPLAY, height: DISPLAY, imageRendering: "pixelated" }}
            aria-label="28 by 28 drawing canvas"
          />
          <div className="mt-4 flex flex-wrap gap-2">
            <button type="button" className="btn-primary" onClick={() => void predict()} disabled={predicting}>
              {predicting ? "Running…" : "Predict"}
            </button>
            <button type="button" className="btn-ghost" onClick={clearCanvas}>
              Clear
            </button>
          </div>
          <p className="muted mt-3">Source: {source}</p>
        </div>

        {samples.length > 0 ? (
          <div className="mt-6 rounded-xl border border-ink-200/80 bg-white/85 p-5 shadow-sm dark:border-white/10 dark:bg-white/[0.04]">
            <h3 className="font-semibold">Or start from a test sample</h3>
            <p className="muted mt-1">Real held-out images, un-normalised for display.</p>
            <div className="mt-3 grid grid-cols-6 gap-2">
              {samples.map((sample) => (
                <SampleButton
                  key={sample.index}
                  sample={sample}
                  onSelect={() => loadPixels(sample.pixels, sample.label_name)}
                />
              ))}
            </div>
          </div>
        ) : null}
      </div>

      <div className="space-y-6 lg:col-span-3">
        <div className="rounded-xl border border-ink-200/80 bg-white/85 p-5 shadow-sm dark:border-white/10 dark:bg-white/[0.04]">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">Prediction</h2>
            {latency !== null ? (
              <span className="text-xs tabular-nums text-ink-700/60 dark:text-ink-100/50">
                {latency.toFixed(1)} ms in the browser
              </span>
            ) : null}
          </div>

          {error ? (
            <div className="mt-4 rounded-lg border border-rose-300/60 bg-rose-50 p-3 text-sm text-rose-800 dark:border-rose-400/30 dark:bg-rose-500/10 dark:text-rose-200">
              <p className="font-medium">Something went wrong</p>
              <p className="mt-1 break-words">{error}</p>
              <button type="button" className="btn-ghost mt-3" onClick={() => void ensureSession()}>
                Retry loading
              </button>
            </div>
          ) : null}

          {!prediction && !error ? (
            <p className="muted mt-3">
              Draw something or pick a sample, then hit Predict. Inference runs locally with ONNX
              Runtime Web (WASM) — nothing is uploaded.
            </p>
          ) : null}

          {prediction ? (
            <div className="mt-4">
              <p className="text-sm text-ink-700/70 dark:text-ink-100/60">Top prediction</p>
              <p className="text-2xl font-semibold">{top?.label}</p>
              <p className="muted">
                {top ? `${(top.probability * 100).toFixed(1)}% confidence` : ""}
              </p>
              <div className="mt-4 space-y-2">
                {prediction.map((item) => (
                  <div key={item.label} className="flex items-center gap-3">
                    <span className="w-32 shrink-0 truncate text-sm">{item.label}</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-100 dark:bg-white/10">
                      <div
                        className={`h-full rounded-full ${
                          item.label === top?.label ? "bg-indigo-500" : "bg-ink-300 dark:bg-white/30"
                        }`}
                        style={{ width: `${Math.max(1, item.probability * 100)}%` }}
                      />
                    </div>
                    <span className="w-14 text-right text-xs tabular-nums">
                      {(item.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>

        <div className="rounded-xl border border-ink-200/80 bg-white/85 p-5 text-sm shadow-sm dark:border-white/10 dark:bg-white/[0.04]">
          <h2 className="font-semibold">Parity note</h2>
          <ul className="muted mt-3 list-disc space-y-2 pl-4 leading-relaxed">
            <li>
              Input is a single grayscale channel at 28×28, scaled to [0, 1] and normalised with
              the training-split mean <span className="font-mono">{mean.toFixed(4)}</span> and
              standard deviation <span className="font-mono">{std.toFixed(4)}</span>. The canvas
              applies exactly the same transform before inference.
            </li>
            <li>
              The ONNX graph takes <span className="font-mono">[batch, 1, 28, 28]</span> and
              returns raw logits; softmax is applied here so the bars sum to 100%.
            </li>
            <li>
              Export used opset 17 with a dynamic batch axis; the exported file is{" "}
              {modelSizeMb.toFixed(2)} MB. PyTorch and ONNX Runtime outputs agreed within{" "}
              <span className="font-mono">{parityDiff.toExponential(1)}</span> max absolute
              difference during the export parity check.
            </li>
            <li>
              Training augmentation (random rotation/translation) is <em>not</em> applied at
              inference — only the deterministic resize/normalise path.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}

function SampleButton({
  sample,
  onSelect,
}: {
  sample: PlaygroundSample;
  onSelect: () => void;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const context = ref.current?.getContext("2d");
    if (!context) return;
    const image = context.createImageData(CANVAS, CANVAS);
    for (let i = 0; i < CANVAS * CANVAS; i += 1) {
      const value = sample.pixels[i] ?? 0;
      image.data[i * 4] = value;
      image.data[i * 4 + 1] = value;
      image.data[i * 4 + 2] = value;
      image.data[i * 4 + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [sample]);

  return (
    <button
      type="button"
      onClick={onSelect}
      title={sample.label_name}
      className="group rounded-md border border-ink-200/70 bg-black p-0.5 transition hover:border-indigo-500 dark:border-white/10"
    >
      <canvas
        ref={ref}
        width={CANVAS}
        height={CANVAS}
        className="block h-full w-full rounded"
        style={{ imageRendering: "pixelated" }}
        aria-label={`Sample of ${sample.label_name}`}
      />
    </button>
  );
}
