import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";

interface ApiErrorStateProps {
  error: Error;
  onRetry: () => void;
  title?: string;
}

/** The standard "API call failed" view -- used everywhere a page's
 * `useApiQuery` status is `"error"`, so offline/failed-request handling
 * looks and behaves the same across the app. */
export function ApiErrorState({ error, onRetry, title = "Couldn't load this data" }: ApiErrorStateProps) {
  return (
    <EmptyState
      icon={AlertTriangle}
      title={title}
      description={error.message}
      action={
        <Button variant="secondary" size="sm" onClick={onRetry}>
          Retry
        </Button>
      }
    />
  );
}
