"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

const TabsContext = React.createContext<{
  value: string;
  onValueChange: (v: string) => void;
} | null>(null);

const Tabs = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & { value?: string; onValueChange?: (v: string) => void; defaultValue?: string }
>(({ className, value, onValueChange, defaultValue, children, ...props }, ref) => {
  const [internal, setInternal] = React.useState(defaultValue ?? "");
  const current = value ?? internal;
  const setCurrent = onValueChange ?? setInternal;
  return (
    <TabsContext.Provider value={{ value: current, onValueChange: setCurrent }}>
      <div ref={ref} className={cn("", className)} {...props}>
        {children}
      </div>
    </TabsContext.Provider>
  );
});
Tabs.displayName = "Tabs";

const TabsList = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
        "inline-flex min-h-12 items-center justify-center gap-1 border border-white/10 bg-black/35 p-1 text-muted-foreground shadow-[inset_0_0_0_1px_rgba(255,255,255,0.03)]",
      className
    )}
    {...props}
  />
));
TabsList.displayName = "TabsList";

const TabsTrigger = React.forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & { value: string }
>(({ className, value, ...props }, ref) => {
  const ctx = React.useContext(TabsContext);
  const active = ctx?.value === value;
  return (
    <button
      ref={ref}
      type="button"
      role="tab"
      aria-selected={active}
      className={cn(
        "inline-flex min-h-10 items-center justify-center whitespace-nowrap border border-transparent px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.22em] text-muted-foreground ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
        active
          ? "border-primary/50 bg-primary/12 text-foreground shadow-[inset_0_0_0_1px_rgba(255,219,180,0.22)]"
          : "hover:border-white/10 hover:bg-white/5 hover:text-foreground",
        className
      )}
      onClick={() => ctx?.onValueChange(value)}
      {...props}
    />
  );
});
TabsTrigger.displayName = "TabsTrigger";

const TabsContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & { value: string }
>(({ className, value, ...props }, ref) => {
  const ctx = React.useContext(TabsContext);
  if (ctx?.value !== value) return null;
  return <div ref={ref} className={cn("mt-2", className)} {...props} />;
});
TabsContent.displayName = "TabsContent";

export { Tabs, TabsList, TabsTrigger, TabsContent };
