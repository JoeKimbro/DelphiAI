"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";

export function StaggerList({
  children,
  delay = 0.05,
  className,
}: {
  children: ReactNode[];
  delay?: number;
  className?: string;
}) {
  return (
    <div className={className}>
      {children.map((child, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.32, delay: i * delay, ease: "easeOut" }}
        >
          {child}
        </motion.div>
      ))}
    </div>
  );
}
