import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] transition-colors",
  {
    variants: {
      variant: {
        default: "border-primary/45 bg-primary/10 text-primary",
        secondary: "border-white/10 bg-white/5 text-muted-foreground",
        outline: "border-white/15 text-foreground",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
