import { Playground } from "@/components/playground";
import { PageHeader } from "@/components/ui";
import { getModelCard, getPlaygroundSamples } from "@/lib/data";

export const dynamic = "force-static";

export const metadata = { title: "Playground" };

const FALLBACK_CLASSES = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
];

export default async function PlaygroundPage() {
  const [samples, card] = await Promise.all([getPlaygroundSamples(), getModelCard()]);
  const classNames = card?.class_names?.length ? card.class_names : FALLBACK_CLASSES;

  return (
    <>
      <PageHeader
        eyebrow="Playground"
        title="Run the trained CNN in your browser"
        description="Draw a garment on the 28×28 canvas or load a real test sample, then run the exported ONNX model locally with onnxruntime-web (WASM). No server inference, no uploads — the same preprocessing steps as training are applied to the canvas pixels."
      />
      <Playground
        samples={samples}
        classNames={classNames}
        mean={card?.dataset.image_mean ?? 0.286}
        std={card?.dataset.image_std ?? 0.353}
        modelSizeMb={card?.onnx.size_mb ?? 0}
        parityDiff={card?.onnx_parity.max_abs_diff ?? 0}
      />
    </>
  );
}
