import type { ReactNode } from "react";

export function EmptyState({
  title,
  description,
  icon,
}: {
  title: string;
  description?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border bg-surface/30 p-10 text-center">
      {icon && <div className="text-muted">{icon}</div>}
      <h3 className="text-lg font-bold text-text">{title}</h3>
      {description && <p className="max-w-md text-sm text-muted">{description}</p>}
    </div>
  );
}
