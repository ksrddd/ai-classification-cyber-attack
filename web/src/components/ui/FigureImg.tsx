"use client";

export function FigureImg({ src, alt }: { src: string; alt: string }) {
  return (
    <img
      src={src}
      alt={alt}
      className="w-full h-auto block rounded"
      onError={(e) => {
        const el = e.target as HTMLImageElement;
        el.style.display = "none";
        const placeholder = document.createElement("div");
        placeholder.className = "py-8 text-center text-[12px] text-[#6C7488]";
        placeholder.innerHTML =
          `<p>Run <code class="font-mono text-[10.5px] py-0.5 px-1 rounded bg-white/[.06] ring-1 ring-white/10 text-[#A8AFC0]">--stage eda</code> to generate this plot.</p>`;
        el.parentElement?.appendChild(placeholder);
      }}
    />
  );
}
