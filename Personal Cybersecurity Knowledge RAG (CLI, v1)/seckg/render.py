from rich.console import Console
from rich.markdown import Markdown
from seckg.models import Answer

console = Console()

def render_answer(answer: Answer):
    """Renders the grounded answer text and bulleted sources nicely using Rich."""
    console.print("\n[bold green]Answer:[/bold green]")
    console.print(Markdown(answer.text))
    
    if answer.sources:
        console.print("\n[bold yellow]Sources:[/bold yellow]")
        seen = set()
        for chunk in answer.sources:
            if chunk.heading_path:
                source_key = f"{chunk.source_file}#{chunk.heading_path}"
                desc = f"[cyan]{chunk.source_file}[/cyan] (Heading: [italic]{chunk.heading_path}[/italic])"
            elif chunk.page_number is not None:
                source_key = f"{chunk.source_file}#page_{chunk.page_number}"
                desc = f"[cyan]{chunk.source_file}[/cyan] (Page {chunk.page_number})"
            else:
                source_key = chunk.source_file
                desc = f"[cyan]{chunk.source_file}[/cyan]"
                
            # Prevent duplicate citation prints
            if source_key not in seen:
                seen.add(source_key)
                console.print(f" • {desc}")
    else:
        console.print("\n[bold red]Sources:[/bold red] None")
    console.print()
