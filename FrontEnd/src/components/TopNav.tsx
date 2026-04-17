import { Link, useLocation } from "react-router-dom";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";

const links = [
  { label: "Dashboard", to: "/" },
  { label: "Simulation", to: "/simulation" },
  { label: "Compare", to: "/compare" },
  { label: "Training", to: "/training" },
  { label: "Communication", to: "/communication" },
];

export function TopNav() {
  const location = useLocation();

  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-14 border-b border-border bg-background/80 backdrop-blur-md flex items-center px-4">
      <SidebarTrigger className="mr-3" />
      <Link to="/" className="text-lg font-bold tracking-wider text-seal-teal mr-8">
        SEAL
      </Link>
      <nav className="hidden md:flex items-center gap-1">
        {links.map((link) => {
          const active = link.to === "/" ? location.pathname === "/" : location.pathname.startsWith(link.to);
          return (
            <Link
              key={link.to}
              to={link.to}
              className={cn(
                "px-3 py-1.5 rounded-md text-sm transition-colors",
                active ? "text-seal-teal bg-muted" : "text-muted-foreground hover:text-foreground"
              )}
            >
              {link.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
