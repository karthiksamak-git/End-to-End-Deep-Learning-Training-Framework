"use client";

import { useEffect, useRef } from "react";

/**
 * Draw a 28x28 grayscale sample (Fashion-MNIST) on a canvas.
 *
 * The pixels come straight from the test split, denormalised by the exporter, so
 * what you see is exactly what the network saw before augmentation.
 */
export function PixelImage({
  pixels,
  size = 84,
  label,
}: {
  pixels: number[];
  size?: number;
  label?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const image = context.createImageData(28, 28);
    for (let i = 0; i < 784; i += 1) {
      const value = pixels[i] ?? 0;
      image.data[i * 4] = value;
      image.data[i * 4 + 1] = value;
      image.data[i * 4 + 2] = value;
      image.data[i * 4 + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [pixels]);

  return (
    <canvas
      ref={canvasRef}
      width={28}
      height={28}
      role="img"
      aria-label={label ?? "28 by 28 pixel sample"}
      className="h-full w-full rounded-md bg-black"
      style={{ imageRendering: "pixelated", width: size, height: size }}
    />
  );
}
