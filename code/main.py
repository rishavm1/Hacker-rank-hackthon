"""
main.py — Entry point for the support triage agent.

Usage:
    python code/main.py

Reads support_tickets/support_tickets.csv, processes each ticket through the
triage pipeline, and writes output to support_tickets/output.csv.
Also writes a detailed reasoning log to the hackathon log file.
"""

import csv
import json
import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add code/ to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

from corpus_loader import load_corpus
from retriever import BM25Retriever
from agent import TriageAgent

# ─── Configuration ───────────────────────────────────────────────────────────

# Load .env if present
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

# Project paths (relative to repo root)
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / 'data'
TICKETS_DIR = REPO_ROOT / 'support_tickets'
INPUT_CSV = TICKETS_DIR / 'support_tickets.csv'
OUTPUT_CSV = TICKETS_DIR / 'output.csv'

# Local execution log
LOCAL_LOG = REPO_ROOT / 'code' / 'execution.log'

console = Console()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger('main')


# ─── Log file helpers ────────────────────────────────────────────────────────

def _append_log(text: str):
    """Append text to the local execution log."""
    with open(LOCAL_LOG, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def _log_ticket(ticket_num: int, total: int, issue: str, subject: str,
                company: str, result: dict, retrieved_chunks: list[str]):
    """Write a detailed log entry for a processed ticket."""
    ist = timezone(timedelta(hours=5, minutes=30))
    timestamp = datetime.now(ist).isoformat()
    
    entry = f"""
## [{timestamp}] TICKET #{ticket_num}/{total}

Classification:
  Status: {result['status']}
  Product Area: {result['product_area']}
  Request Type: {result['request_type']}

Retrieved chunks:
{chr(10).join(f'  - {c}' for c in retrieved_chunks[:3])}

---
"""
    _append_log(entry)


# ─── CSV I/O ─────────────────────────────────────────────────────────────────

def read_tickets(path: Path) -> list[dict]:
    """Read support tickets from CSV."""
    tickets = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize field names
            ticket = {
                'issue': (row.get('Issue', '') or '').strip(),
                'subject': (row.get('Subject', '') or '').strip(),
                'company': (row.get('Company', '') or '').strip(),
            }
            # Skip completely empty rows
            if ticket['issue'] or ticket['subject']:
                tickets.append(ticket)
    return tickets


def write_output(results: list[dict], path: Path):
    """Write processed results to output CSV."""
    fieldnames = ['issue', 'subject', 'company', 'response', 'product_area',
                  'status', 'request_type', 'justification']
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in results:
            # Validate output fields strictly
            row = {}
            for k in fieldnames:
                val = str(r.get(k, ''))
                # Strictly remove newlines/returns to ensure single-line
                val = val.replace('\n', ' ').replace('\r', ' ').strip()
                # Remove double spaces
                val = " ".join(val.split())
                row[k] = val
            
            # Ensure status is valid
            if row['status'] not in ('replied', 'escalated'):
                row['status'] = 'escalated'
            
            writer.writerow(row)


# ─── Main pipeline ──────────────────────────────────────────────────────────

def main():
    """Run the full triage pipeline."""
    ist = timezone(timedelta(hours=5, minutes=30))
    start_time = time.time()
    
    console.print(Panel.fit(
        "[bold cyan]🤖 Multi-Domain Support Triage Agent[/bold cyan]\n"
        "[dim]HackerRank Orchestrate — May 2026[/dim]",
        border_style="cyan",
    ))
    
    # ── 1. Log session start ──────────────────────────────────────────
    _append_log(f"""
## [{datetime.now(ist).isoformat()}] SESSION START — Agent Run

Agent: support-triage-agent
Repo Root: {REPO_ROOT}
Language: py
Time: {datetime.now(ist).isoformat()}
""")
    
    # ── 2. Check API key ──────────────────────────────────────────────
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        console.print("[bold red]❌ OPENROUTER_API_KEY not set![/bold red]")
        console.print("Set it with: $env:OPENROUTER_API_KEY = 'your-key-here'")
        sys.exit(1)
    console.print("[green]✓[/green] API key found")
    
    # ── 3. Load corpus ────────────────────────────────────────────────
    with console.status("[bold yellow]Loading support corpus...[/bold yellow]"):
        documents = load_corpus(str(DATA_DIR))
    
    corpus_table = Table(title="📚 Corpus Summary", show_header=True)
    corpus_table.add_column("Domain", style="cyan")
    corpus_table.add_column("Documents", style="green", justify="right")
    for domain in ['hackerrank', 'claude', 'visa']:
        count = sum(1 for d in documents if d['domain'] == domain)
        corpus_table.add_row(domain.title(), str(count))
    corpus_table.add_row("[bold]Total[/bold]", f"[bold]{len(documents)}[/bold]")
    console.print(corpus_table)
    
    # ── 4. Build retriever ────────────────────────────────────────────
    with console.status("[bold yellow]Building TF-IDF index...[/bold yellow]"):
        retriever = BM25Retriever(documents)
    console.print(f"[green]✓[/green] TF-IDF index built ({len(documents)} docs)")
    
    # ── 5. Initialize agent ───────────────────────────────────────────
    agent = TriageAgent(retriever, api_key=api_key)
    console.print(f"[green]✓[/green] Agent initialized (model: {agent.model})")
    
    # ── 6. Load tickets ───────────────────────────────────────────────
    tickets = read_tickets(INPUT_CSV)
    console.print(f"[green]✓[/green] Loaded {len(tickets)} tickets from {INPUT_CSV.name}")
    
    # ── 7. Process tickets ────────────────────────────────────────────
    results = [None] * len(tickets)
    console.print()
    
    def process_single_ticket(idx, ticket):
        try:
            result = agent.process_ticket(
                ticket['issue'],
                ticket['subject'],
                ticket['company'],
            )
            # Log the ticket
            retrieved = retriever.retrieve(
                f"{ticket['issue']} {ticket['subject']}",
                domain=None, top_k=3
            )
            chunk_titles = [r['document']['title'][:80] for r in retrieved]
            _log_ticket(idx + 1, len(tickets), ticket['issue'],
                       ticket['subject'], ticket['company'],
                       result, chunk_titles)
            return idx, result
        except Exception as e:
            logger.error(f"Error processing ticket {idx+1}: {e}")
            result = {
                'issue': ticket['issue'],
                'subject': ticket['subject'],
                'company': ticket['company'],
                'response': 'This issue requires human review due to a processing error.',
                'product_area': 'general_support',
                'status': 'escalated',
                'request_type': 'product_issue',
                'justification': f'Processing error: {str(e)[:200]}',
            }
            return idx, result

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing tickets...", total=len(tickets))
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_single_ticket, i, t): i for i, t in enumerate(tickets)}
            for future in as_completed(futures):
                idx = futures[future]
                _, result = future.result()
                results[idx] = result
                progress.advance(task)
    
    # ── 8. Write output ───────────────────────────────────────────────
    write_output(results, OUTPUT_CSV)
    console.print(f"\n[green]✓[/green] Output written to [bold]{OUTPUT_CSV}[/bold]")
    
    # ── 9. Summary ────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    
    summary_table = Table(title="📊 Results Summary", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")
    
    replied = sum(1 for r in results if r['status'] == 'replied')
    escalated = sum(1 for r in results if r['status'] == 'escalated')
    summary_table.add_row("Total Tickets", str(len(results)))
    summary_table.add_row("Replied", str(replied))
    summary_table.add_row("Escalated", str(escalated))
    summary_table.add_row("Time", f"{elapsed:.1f}s")
    
    # Request type breakdown
    for rt in ['product_issue', 'feature_request', 'bug', 'invalid']:
        count = sum(1 for r in results if r['request_type'] == rt)
        if count > 0:
            summary_table.add_row(f"  {rt}", str(count))
    
    console.print(summary_table)
    
    # Final log
    _append_log(f"""
## [{datetime.now(ist).isoformat()}] RUN COMPLETE

Total tickets: {len(results)}
Replied: {replied}
Escalated: {escalated}
Time: {elapsed:.1f}s
Output: {OUTPUT_CSV}
""")
    
    console.print(f"\n[dim]Log written to {LOCAL_LOG}[/dim]")
    console.print("[bold green]✅ Done![/bold green]")


if __name__ == '__main__':
    main()
