import { useEffect, useRef } from "react";
import { ModelState } from "../App";

interface RandomForestVisualizationProps {
  trees: number;
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  modelState: ModelState;
  onTreesChange?: (value: number) => void;
  onMaxDepthChange?: (value: number) => void;
  onMinSamplesSplitChange?: (value: number) => void;
  onMinSamplesLeafChange?: (value: number) => void;
}

interface TreeNode {
  x: number;
  y: number;
  depth: number;
  isLeaf: boolean;
  children: TreeNode[];
  brightness: number;
  pulse: number;
}

export function RandomForestVisualization({
  trees,
  maxDepth,
  minSamplesSplit,
  minSamplesLeaf,
  modelState,
  onTreesChange,
  onMaxDepthChange,
  onMinSamplesSplitChange,
  onMinSamplesLeafChange,
}: RandomForestVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { isTrained, isTraining } = modelState;

  // Validation functions
  const validateTrees = (value: number) => Math.max(10, Math.min(500, value));
  const validateMaxDepth = (value: number) => Math.max(3, Math.min(30, value));
  const validateMinSamplesSplit = (value: number) => Math.max(2, Math.min(20, value));
  const validateMinSamplesLeaf = (value: number) => Math.max(1, Math.min(10, value));

  // Input handlers with validation
  const handleTreesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || 10;
    if (onTreesChange) onTreesChange(validateTrees(value));
  };

  const handleMaxDepthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || 3;
    if (onMaxDepthChange) onMaxDepthChange(validateMaxDepth(value));
  };

  const handleMinSamplesSplitChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || 2;
    if (onMinSamplesSplitChange) onMinSamplesSplitChange(validateMinSamplesSplit(value));
  };

  const handleMinSamplesLeafChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || 1;
    if (onMinSamplesLeafChange) onMinSamplesLeafChange(validateMinSamplesLeaf(value));
  };

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

    // Calculate how many trees to display (show representative sample, max 5)
    const displayTrees = Math.min(5, Math.max(2, Math.ceil(trees / 100)));

    // Calculate visible depth based on maxDepth parameter (cap at 5 for visual clarity)
    const visibleDepth = Math.min(5, maxDepth);

    // Calculate node size based on minSamplesLeaf
    const nodeSize = Math.max(4, Math.min(10, 12 - minSamplesLeaf));

    // Generate tree structures
    const allTrees: TreeNode[][] = [];
    const treeWidth = width / displayTrees;

    for (let t = 0; t < displayTrees; t++) {
      const treeNodes: TreeNode[] = [];
      const treeOffsetX = t * treeWidth + treeWidth / 2;
      const verticalSpacing = Math.min(80, (height - 80) / (visibleDepth + 1));

      // Build binary tree structure
      const buildTree = (
        x: number,
        y: number,
        depth: number,
        horizontalSpread: number,
      ): TreeNode | null => {
        if (depth > visibleDepth) return null;

        // Probability of being a leaf increases with depth and minSamplesSplit
        const leafProbability = (depth / visibleDepth) * (minSamplesSplit / 20);
        const isLeaf =
          depth === visibleDepth ||
          (depth > 1 && Math.random() < leafProbability);

        const node: TreeNode = {
          x,
          y,
          depth,
          isLeaf,
          children: [],
          brightness: 0.6 + Math.random() * 0.4,
          pulse: Math.random() * Math.PI * 2,
        };

        treeNodes.push(node);

        if (!isLeaf) {
          const nextY = y + verticalSpacing;
          const nextSpread = horizontalSpread * 0.55;

          const leftChild = buildTree(
            x - nextSpread,
            nextY,
            depth + 1,
            nextSpread,
          );
          const rightChild = buildTree(
            x + nextSpread,
            nextY,
            depth + 1,
            nextSpread,
          );

          if (leftChild) node.children.push(leftChild);
          if (rightChild) node.children.push(rightChild);
        }

        return node;
      };

      const initialSpread = treeWidth * 0.35;
      buildTree(treeOffsetX, 60, 0, initialSpread);
      allTrees.push(treeNodes);
    }

    // Animation
    let animationFrame: number;
    let time = 0;

    const animate = () => {
      ctx.clearRect(0, 0, width, height);
      time += 0.01;

      // Draw each tree
      allTrees.forEach((treeNodes, treeIndex) => {
        // Draw connections (branches)
        treeNodes.forEach((node) => {
          node.children.forEach((child) => {
            const gradient = ctx.createLinearGradient(
              node.x,
              node.y,
              child.x,
              child.y,
            );
            gradient.addColorStop(0, `rgba(57, 255, 20, 0.4)`);
            gradient.addColorStop(1, `rgba(57, 255, 20, 0.2)`);

            ctx.strokeStyle = gradient;
            ctx.lineWidth = Math.max(1, 3 - node.depth * 0.5);
            ctx.beginPath();
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(child.x, child.y);
            ctx.stroke();
          });
        });

        // Draw nodes
        treeNodes.forEach((node) => {
          const pulse = Math.sin(time * 2 + node.pulse) * 0.15 + 0.85;
          const brightness = node.brightness * pulse;
          const size = nodeSize * (node.isLeaf ? 0.8 : 1);

          // Outer glow
          const glowGradient = ctx.createRadialGradient(
            node.x,
            node.y,
            0,
            node.x,
            node.y,
            size * 2.5,
          );

          if (node.isLeaf) {
            // Leaf nodes - slightly different color (more blue-green)
            glowGradient.addColorStop(
              0,
              `rgba(20, 255, 150, ${brightness * 0.4})`,
            );
            glowGradient.addColorStop(
              0.5,
              `rgba(20, 255, 150, ${brightness * 0.15})`,
            );
            glowGradient.addColorStop(1, "rgba(20, 255, 150, 0)");
          } else {
            // Decision nodes
            glowGradient.addColorStop(
              0,
              `rgba(57, 255, 20, ${brightness * 0.5})`,
            );
            glowGradient.addColorStop(
              0.5,
              `rgba(57, 255, 20, ${brightness * 0.2})`,
            );
            glowGradient.addColorStop(1, "rgba(57, 255, 20, 0)");
          }

          ctx.fillStyle = glowGradient;
          ctx.beginPath();
          ctx.arc(node.x, node.y, size * 2.5, 0, Math.PI * 2);
          ctx.fill();

          // Node shape - diamonds for decision nodes, circles for leaves
          ctx.strokeStyle = node.isLeaf
            ? `rgba(20, 255, 150, ${brightness * 0.9})`
            : `rgba(57, 255, 20, ${brightness * 0.9})`;
          ctx.lineWidth = 2;

          if (node.isLeaf) {
            // Circle for leaf
            ctx.beginPath();
            ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
            ctx.stroke();
            ctx.fillStyle = "rgba(10, 10, 15, 0.9)";
            ctx.fill();
          } else {
            // Diamond for decision node
            ctx.beginPath();
            ctx.moveTo(node.x, node.y - size * 1.2);
            ctx.lineTo(node.x + size * 1.2, node.y);
            ctx.lineTo(node.x, node.y + size * 1.2);
            ctx.lineTo(node.x - size * 1.2, node.y);
            ctx.closePath();
            ctx.stroke();
            ctx.fillStyle = "rgba(10, 10, 15, 0.9)";
            ctx.fill();
          }

          // Inner highlight
          ctx.fillStyle = node.isLeaf
            ? `rgba(20, 255, 150, ${brightness * 0.5})`
            : `rgba(57, 255, 20, ${brightness * 0.5})`;
          ctx.beginPath();
          ctx.arc(node.x, node.y, size * 0.35, 0, Math.PI * 2);
          ctx.fill();
        });

        // Draw data flow animation when training or trained
        if (isTraining || isTrained) {
          const flowProgress = (time * 0.5 + treeIndex * 0.3) % 1;
          const currentDepth = Math.floor(flowProgress * (visibleDepth + 1));

          // Find nodes at current depth to highlight
          treeNodes.forEach((node) => {
            if (node.depth === currentDepth) {
              const alpha =
                0.6 *
                (1 -
                  Math.abs(
                    flowProgress * (visibleDepth + 1) - currentDepth - 0.5,
                  ));

              ctx.fillStyle = `rgba(255, 255, 100, ${alpha})`;
              ctx.beginPath();
              ctx.arc(node.x, node.y, nodeSize * 0.5, 0, Math.PI * 2);
              ctx.fill();
            }
          });
        }
      });

      // Draw tree labels directly under each tree
      ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
      ctx.font = "10px Inter, sans-serif";
      ctx.textAlign = "center";

      for (let t = 0; t < displayTrees; t++) {
        const treeNodes = allTrees[t];
        // Find the maximum y position (deepest node) in this tree
        const maxY = Math.max(...treeNodes.map((node) => node.y));
        const x = t * treeWidth + treeWidth / 2;
        ctx.fillText(`Tree ${t + 1}`, x, maxY + 40);
      }

      // Draw tree count indicator
      if (trees > displayTrees) {
        ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
        ctx.font = "11px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(
          `Showing ${displayTrees} of ${trees} trees`,
          width / 2,
          25,
        );
      }

      animationFrame = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationFrame);
    };
  }, [trees, maxDepth, minSamplesSplit, minSamplesLeaf, isTraining, isTrained]);

  return (
    <div className="h-full w-full flex flex-col">
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

      {/* Model Parameters - Only show before training */}
      {!isTrained && (
        <div className="mb-3 flex justify-center gap-3 text-xs">
          <div className="flex items-center gap-2 px-2 py-1 rounded bg-white/5">
            <input
              type="number"
              value={trees}
              onChange={handleTreesChange}
              disabled={isTraining}
              min="10"
              max="500"
              className="w-12 bg-transparent text-gray-400 border-none outline-none focus:text-white focus:ring-1 focus:ring-[#39FF14]/50 rounded text-center disabled:opacity-50"
            />
            <span className="text-gray-400">Trees</span>
          </div>
          <div className="flex items-center gap-2 px-2 py-1 rounded bg-white/5">
            <input
              type="number"
              value={maxDepth}
              onChange={handleMaxDepthChange}
              disabled={isTraining}
              min="3"
              max="30"
              className="w-8 bg-transparent text-gray-400 border-none outline-none focus:text-white focus:ring-1 focus:ring-[#39FF14]/50 rounded text-center disabled:opacity-50"
            />
            <span className="text-gray-400">Max Depth</span>
          </div>
          <div className="flex items-center gap-2 px-2 py-1 rounded bg-white/5">
            <input
              type="number"
              value={minSamplesSplit}
              onChange={handleMinSamplesSplitChange}
              disabled={isTraining}
              min="2"
              max="20"
              className="w-8 bg-transparent text-gray-400 border-none outline-none focus:text-white focus:ring-1 focus:ring-[#39FF14]/50 rounded text-center disabled:opacity-50"
            />
            <span className="text-gray-400">Min Split</span>
          </div>
          <div className="flex items-center gap-2 px-2 py-1 rounded bg-white/5">
            <input
              type="number"
              value={minSamplesLeaf}
              onChange={handleMinSamplesLeafChange}
              disabled={isTraining}
              min="1"
              max="10"
              className="w-8 bg-transparent text-gray-400 border-none outline-none focus:text-white focus:ring-1 focus:ring-[#39FF14]/50 rounded text-center disabled:opacity-50"
            />
            <span className="text-gray-400">Min Leaf</span>
          </div>
        </div>
      )}

      {/* Canvas Visualization */}
      <div className="flex-1 relative rounded-xl overflow-hidden bg-gradient-to-br from-[#0A0A0F] to-[#14141F] border border-white/10">
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ width: "100%", height: "100%" }}
        />

        {/* Legend */}
        <div className="absolute top-3 left-3 flex gap-4 text-xs text-gray-400">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rotate-45 border border-[#39FF14]/70 bg-[#0A0A0F]"></div>
            <span>Decision Node</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-full border border-[#14FF96]/70 bg-[#0A0A0F]"></div>
            <span>Leaf Node</span>
          </div>
        </div>
      </div>
    </div>
  );
}
