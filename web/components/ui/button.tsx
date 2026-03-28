import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap border border-transparent px-4 text-[11px] font-semibold uppercase tracking-[0.24em] transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "border-primary/70 bg-primary/95 text-primary-foreground shadow-[0_0_0_1px_rgba(255,220,183,0.18)] hover:-translate-y-px hover:brightness-105",
        secondary:
          "border-white/10 bg-[hsl(var(--panel-elevated))] text-foreground hover:-translate-y-px hover:border-primary/35 hover:bg-white/6",
        outline:
          "border-white/15 bg-transparent text-foreground hover:-translate-y-px hover:border-primary/80 hover:bg-primary/10",
        ghost:
          "border-transparent bg-transparent text-muted-foreground hover:border-white/10 hover:bg-white/5 hover:text-foreground",
        link: "h-auto border-none px-0 text-primary underline-offset-4 hover:text-white hover:underline",
      },
      size: {
        default: "h-11",
        sm: "h-9 px-3 text-[10px]",
        lg: "h-12 px-8 text-xs",
        icon: "h-11 w-11 px-0",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      className={cn(buttonVariants({ variant, size, className }))}
      ref={ref}
      {...props}
    />
  )
);
Button.displayName = "Button";

export { Button, buttonVariants };
