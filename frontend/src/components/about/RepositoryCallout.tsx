import { Check, FolderGit2 } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

const REPO_URL = "https://github.com/poojanegi17/RL-Minesweeper-Lab";

const HIGHLIGHTS = [
  "Modular agents -- each algorithm is an independent, swappable implementation behind one shared environment interface",
  "Reproducible experiments -- seeded runs with recorded configuration, not one-off scripts",
  "Documented decisions -- why each technique was introduced, not just that it was",
  "A real evaluation pipeline -- training and evaluation are separate passes, not the same run reported twice",
];

/** Closing CTA -- what a reviewer will actually find in the repository,
 * not a generic "check it out" line. */
export function RepositoryCallout() {
  return (
    <Card className="flex flex-col gap-6 border-white/10 bg-gradient-to-b from-surface/80 to-surface/50 shadow-lg shadow-black/[0.06] backdrop-blur-sm sm:flex-row sm:items-center sm:justify-between">
      <div className="flex flex-col gap-3">
        <h2 className="text-xl font-semibold text-heading">Explore the code</h2>
        <ul className="flex flex-col gap-2">
          {HIGHLIGHTS.map((highlight) => (
            <li key={highlight} className="flex items-start gap-2 text-sm text-text-muted">
              <Check className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
              {highlight}
            </li>
          ))}
        </ul>
      </div>

      <a href={REPO_URL} target="_blank" rel="noreferrer" className="shrink-0">
        <Button>
          <FolderGit2 className="h-4 w-4" />
          View on GitHub
        </Button>
      </a>
    </Card>
  );
}
