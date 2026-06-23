"use client";

// The "Elsewhere" wordmark shown top-left on every screen. Kept in one place so
// the brand looks identical everywhere and consistently acts as the "go home"
// affordance. When `onClick` is provided it renders as a button (with hover +
// focus styles); otherwise it stays a plain label. `showDot` matches the small
// gradient dot used during a live session.
export function BrandHome({
  onClick,
  showDot = false,
  ariaLabel = "Go to home",
}: {
  onClick?: () => void;
  showDot?: boolean;
  ariaLabel?: string;
}) {
  const content = (
    <span className="flex items-center gap-2.5">
      {showDot ? (
        <span className="h-3.5 w-3.5 rounded-full bg-gradient-to-br from-[#E0C9A8] to-[#C4A882]" />
      ) : null}
      <span className="text-sm font-light tracking-tight text-white/85">Elsewhere</span>
    </span>
  );

  if (!onClick) {
    return <div className="flex items-center">{content}</div>;
  }

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      className="-m-1 flex items-center rounded-md p-1 transition-opacity hover:opacity-80 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/40"
    >
      {content}
    </button>
  );
}
