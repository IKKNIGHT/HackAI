import { useEffect, useRef } from "react";
import { ModelState } from "../App";

interface ModelVisualizationProps {
  trees: number;
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  modelState: ModelState;
}

export function ModelVisualization({
  trees,
  maxDepth,
  minSamplesSplit,
  minSamplesLeaf,
  modelState,
}: ModelVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { isTrained, isTraining } = modelState;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    // Calculate nodes per layer based on trees parameter (10-500 maps to 3-7 nodes)
    const nodesPerLayer = Math.max(
      3,
      Math.min(7, Math.floor(3 + (trees - 10) / 80)),
    );

    // Calculate number of hidden layers based on maxDepth (3-30 maps to 1-4 hidden layers)
    const hiddenLayers = Math.max(
      1,
      Math.min(4, Math.floor((maxDepth - 3) / 7) + 1),
    );

    // Create neural network layers structure
    const layers: Array<{ name: string; nodeCount: number }> = [
      { name: "Input", nodeCount: nodesPerLayer },
    ];

    // Add hidden layers
    for (let i = 0; i < hiddenLayers; i++) {
      layers.push({ name: `Hidden ${i + 1}`, nodeCount: nodesPerLayer });
    }

    layers.push({ name: "Output", nodeCount: 2 });

    const nodes: Array<{
      x: number;
      y: number;
      layer: number;
      brightness: number;
      pulse: number;
    }> = [];
    const layerSpacing = width / (layers.length + 1);

    layers.forEach((layer, layerIndex) => {
      const x = layerSpacing * (layerIndex + 1);
      const verticalSpacing = height / (layer.nodeCount + 1);

      for (let i = 0; i < layer.nodeCount; i++) {
        const y = verticalSpacing * (i + 1);
        nodes.push({
          x,
          y,
          layer: layerIndex,
          brightness: 0.6 + Math.random() * 0.4,
          pulse: Math.random() * Math.PI * 2,
        });
      }
    });

    // Calculate connection opacity based on minSamplesSplit (2-20 maps to more/less connections visible)
    const connectionAlpha = 0.05 + (0.25 * (20 - minSamplesSplit)) / 18;

    // Animation
    let animationFrame: number;
    let time = 0;

    const animate = () => {
      ctx.clearRect(0, 0, width, height);
      time += 0.008; // Slower animation speed

      // Draw all connections between adjacent layers
      nodes.forEach((node, i) => {
        nodes.forEach((targetNode, j) => {
          if (targetNode.layer === node.layer + 1) {
            ctx.strokeStyle = `rgba(57, 255, 20, ${connectionAlpha})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(targetNode.x, targetNode.y);
            ctx.stroke();
          }
        });
      });

      // Draw nodes with size affected by minSamplesLeaf
      const baseSize = Math.max(8, 14 - minSamplesLeaf);

      nodes.forEach((node, i) => {
        const pulse = Math.sin(time + node.pulse) * 0.1 + 0.9; // Subtler pulsing
        const brightness = node.brightness * pulse;
        const size = baseSize;

        // Outer glow
        const gradient = ctx.createRadialGradient(
          node.x,
          node.y,
          0,
          node.x,
          node.y,
          size * 2,
        );
        gradient.addColorStop(0, `rgba(57, 255, 20, ${brightness * 0.5})`);
        gradient.addColorStop(0.5, `rgba(57, 255, 20, ${brightness * 0.2})`);
        gradient.addColorStop(1, "rgba(57, 255, 20, 0)");

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(node.x, node.y, size * 2, 0, Math.PI * 2);
        ctx.fill();

        // Main circle with border
        ctx.strokeStyle = `rgba(57, 255, 20, ${brightness * 0.9})`;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
        ctx.stroke();

        // Inner fill
        ctx.fillStyle = "rgba(10, 10, 15, 0.8)";
        ctx.fill();

        // Bright center dot
        ctx.fillStyle = `rgba(57, 255, 20, ${brightness * 0.6})`;
        ctx.beginPath();
        ctx.arc(node.x, node.y, size * 0.4, 0, Math.PI * 2);
        ctx.fill();
      });

      // Flowing light effects on random connections
      const flowCount = 3; // Fewer flowing lights
      for (let f = 0; f < flowCount; f++) {
        const maxLayer = layers.length - 2;
        const sourceLayer = Math.floor((time * 0.3 + f * 1.7) % maxLayer);
        const sourceNodes = nodes.filter((n) => n.layer === sourceLayer);
        const targetNodes = nodes.filter((n) => n.layer === sourceLayer + 1);

        if (sourceNodes.length > 0 && targetNodes.length > 0) {
          const sourceIndex =
            Math.floor((time * 1.5 + f * 2.3) * 10) % sourceNodes.length;
          const targetIndex =
            Math.floor((time * 1.2 + f * 1.9) * 10) % targetNodes.length;
          const source = sourceNodes[sourceIndex];
          const target = targetNodes[targetIndex];

          const progress = (time * 1 + f * 0.7) % 1; // Slower flow speed
          const x = source.x + (target.x - source.x) * progress;
          const y = source.y + (target.y - source.y) * progress;

          const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
          gradient.addColorStop(0, "rgba(57, 255, 20, 0.8)");
          gradient.addColorStop(0.5, "rgba(57, 255, 20, 0.4)");
          gradient.addColorStop(1, "rgba(57, 255, 20, 0)");

          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(x, y, 8, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      animationFrame = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationFrame);
    };
  }, [trees, maxDepth, minSamplesSplit, minSamplesLeaf]);

  return (
    <div className="h-full flex flex-col">
      {/* Badge */}
      <div
        className={`mb-4 inline-flex items-center gap-2 px-4 py-2 rounded-full self-center ${
          isTraining
            ? "bg-yellow-500/10 border border-yellow-500/50 shadow-[0_0_20px_rgba(234,179,8,0.3)]"
            : isTrained
              ? "bg-[#39FF14]/10 border border-[#39FF14]/50 shadow-[0_0_20px_rgba(57,255,20,0.3)]"
              : "bg-white/5 border border-white/20"
        }`}
      >
        <div
          className={`w-2 h-2 rounded-full ${
            isTraining
              ? "bg-yellow-500 animate-pulse"
              : isTrained
                ? "bg-[#39FF14] animate-pulse"
                : "bg-gray-500"
          }`}
        ></div>
        <span
          className={`text-sm font-semibold ${
            isTraining
              ? "text-yellow-500"
              : isTrained
                ? "text-[#39FF14]"
                : "text-gray-400"
          }`}
        >
          {isTraining
            ? "TRAINING IN PROGRESS..."
            : isTrained
              ? "RANDOM FOREST ACTIVE"
              : "NO MODEL TRAINED"}
        </span>
      </div>

      {/* Canvas Visualization */}
      <div className="flex-1 relative rounded-xl overflow-hidden bg-gradient-to-br from-[#0A0A0F] to-[#14141F] border border-white/10">
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ width: "100%", height: "100%" }}
        />
      </div>
    </div>
  );
}
