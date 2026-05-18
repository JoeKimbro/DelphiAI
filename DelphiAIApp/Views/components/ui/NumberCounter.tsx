"use client";

import { useEffect, useRef } from "react";
import { animate, useInView, useMotionValue, useTransform, motion } from "framer-motion";

export function NumberCounter({
  value,
  decimals = 0,
  suffix = "",
  prefix = "",
  duration = 0.9,
  className,
}: {
  value: number;
  decimals?: number;
  suffix?: string;
  prefix?: string;
  duration?: number;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const mv = useMotionValue(0);
  const rounded = useTransform(mv, (v) =>
    `${prefix}${v.toFixed(decimals)}${suffix}`
  );

  useEffect(() => {
    if (inView) {
      const controls = animate(mv, value, { duration, ease: "easeOut" });
      return () => controls.stop();
    }
  }, [inView, mv, value, duration]);

  return (
    <motion.span ref={ref} className={className}>
      {rounded}
    </motion.span>
  );
}
