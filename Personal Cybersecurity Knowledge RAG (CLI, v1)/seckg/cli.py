import os
import sys
import time
import click

from seckg.config import settings
from seckg.ingest import ingest_directory
from seckg.vector_store import VectorStoreManager
from seckg.retriever import Retriever
from seckg.answerer import Answerer
from seckg.render import render_answer, console
from seckg.watcher import DirectoryWatcher

@click.group()
def cli():
    """SecKnowledge - A personal RAG system for cybersecurity notes."""
    pass

@cli.command("index")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
def index_cmd(path):
    """Initial full index of a directory."""
    abs_path = os.path.abspath(path)
    console.print(f"[bold blue]Initializing indexing for directory:[/bold blue] {abs_path}")
    
    with console.status("[bold green]Loading documents and generating chunks..."):
        try:
            # Full scan and ingestion of documents
            chunks = ingest_directory(abs_path)
        except Exception as e:
            console.print(f"[bold red]Ingestion error:[/bold red] {e}")
            sys.exit(1)
            
    if not chunks:
        console.print("[yellow]No markdown (.md) or PDF (.pdf) files found to index.[/yellow]")
        return
        
    console.print(f"Found and chunked [green]{len(chunks)}[/green] sections.")
    
    with console.status("[bold green]Generating embeddings and persisting database..."):
        try:
            manager = VectorStoreManager()
            # Clear collection for clean start on full rebuild
            manager.reset_store()
            manager.add_chunks(chunks)
        except Exception as e:
            console.print(f"[bold red]Indexing error:[/bold red] {e}")
            sys.exit(1)
            
    console.print("[bold green]Success![/bold green] Database index built and persisted.")

@cli.command("watch")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
def watch_cmd(path):
    """Start live file-watcher for incremental re-indexing."""
    abs_path = os.path.abspath(path)
    console.print(f"[bold blue]Starting file watcher on:[/bold blue] {abs_path}")
    console.print("Press Ctrl+C to stop.")
    
    try:
        manager = VectorStoreManager()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize vector store:[/bold red] {e}")
        sys.exit(1)
        
    def log_callback(msg):
        current_time = time.strftime("%H:%M:%S")
        console.print(f"[{current_time}] [bold green]Watcher:[/bold green] {msg}")

    watcher = DirectoryWatcher(abs_path, manager, callback=log_callback)
    watcher.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Stopping file watcher...[/bold yellow]")
    finally:
        watcher.stop()
        console.print("[bold green]File watcher stopped.[/bold green]")

@cli.command("ask")
@click.argument("question")
@click.option("--k", default=None, type=int, help="Override retrieval count.")
def ask_cmd(question, k):
    """Query the RAG system (one-shot)."""
    if not settings.gemini_api_key:
        console.print("[bold red]Error:[/bold red] GEMINI_API_KEY is not set. Please set it in your .env file.")
        sys.exit(1)
        
    try:
        manager = VectorStoreManager()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize vector store:[/bold red] {e}")
        sys.exit(1)
        
    retriever = Retriever(manager)
    answerer = Answerer()
    
    with console.status("[bold green]Searching knowledge base & generating answer..."):
        results = retriever.retrieve(question, k=k)
        answer = answerer.answer(question, results)
        
    render_answer(answer)

@cli.command("chat")
@click.option("--k", default=None, type=int, help="Override retrieval count.")
def chat_cmd(k):
    """Start an interactive chat session."""
    if not settings.gemini_api_key:
        console.print("[bold red]Error:[/bold red] GEMINI_API_KEY is not set. Please set it in your .env file.")
        sys.exit(1)
        
    try:
        manager = VectorStoreManager()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize vector store:[/bold red] {e}")
        sys.exit(1)
        
    retriever = Retriever(manager)
    answerer = Answerer()
    
    console.print("[bold green]Welcome to SecKnowledge interactive chat![/bold green]")
    console.print("Type your questions below. Type 'exit' or 'quit' to end the session.")
    console.print("-" * 50)
    
    while True:
        try:
            question = console.input("[bold cyan]seckg>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Ending chat session. Goodbye![/bold yellow]")
            break
            
        if not question:
            continue
            
        if question.lower() in ["exit", "quit"]:
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break
            
        with console.status("[bold green]Searching & generating answer..."):
            results = retriever.retrieve(question, k=k)
            answer = answerer.answer(question, results)
            
        render_answer(answer)

if __name__ == "__main__":
    cli()
