import os
import time
import threading
from typing import Dict, Callable, Optional
# pyrefly: ignore [missing-import]
from watchdog.observers import Observer
# pyrefly: ignore [missing-import]
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from seckg.config import settings
from seckg.ingest import ingest_file
from seckg.vector_store import VectorStoreManager

class DebouncedReindexHandler(FileSystemEventHandler):
    """Handles file system events with debouncing to prevent excessive updates on save."""
    
    def __init__(self, manager: VectorStoreManager, base_dir: str, callback: Optional[Callable[[str], None]] = None):
        self.manager = manager
        self.base_dir = os.path.abspath(base_dir)
        self.callback = callback
        self.debounce_seconds = settings.watcher_debounce_seconds
        
        # Maps relative file path -> scheduled timestamp to run reindexing
        self.pending_events: Dict[str, float] = {}
        self.lock = threading.Lock()
        
        # Start worker thread for processing debounced events
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def on_any_event(self, event: FileSystemEvent):
        # Ignore directory changes, only handle files
        if event.is_directory:
            return
            
        file_path = os.path.abspath(event.src_path)
        
        # Only monitor markdown and PDF files
        _, ext = os.path.splitext(file_path.lower())
        if ext not in [".md", ".pdf"]:
            return
            
        # Ignore hidden files and directories (like .obsidian, .git, etc.)
        rel_path = os.path.relpath(file_path, self.base_dir)
        if any(part.startswith(".") for part in rel_path.split(os.sep)):
            return

        with self.lock:
            # Set or refresh the run time for this file path
            self.pending_events[rel_path] = time.time() + self.debounce_seconds

    def _worker(self):
        """Monitors scheduled events and processes them after the debounce period has elapsed."""
        while self.running:
            time.sleep(0.2)
            now = time.time()
            to_process = []
            
            with self.lock:
                for rel_path, target_time in list(self.pending_events.items()):
                    if now >= target_time:
                        to_process.append(rel_path)
                        del self.pending_events[rel_path]
                        
            for rel_path in to_process:
                abs_path = os.path.join(self.base_dir, rel_path)
                try:
                    if os.path.exists(abs_path):
                        # File created or modified
                        # Delete existing entries in Chroma DB first to avoid duplicate chunks
                        self.manager.delete_by_file(rel_path)
                        # Load and chunk the updated file
                        chunks = ingest_file(abs_path, self.base_dir)
                        if chunks:
                            self.manager.add_chunks(chunks)
                            msg = f"Re-indexed: {rel_path} ({len(chunks)} chunks)"
                        else:
                            msg = f"Skipped indexing for empty file: {rel_path}"
                    else:
                        # File deleted
                        self.manager.delete_by_file(rel_path)
                        msg = f"Removed index entries for deleted file: {rel_path}"
                        
                    if self.callback:
                        self.callback(msg)
                    else:
                        print(msg)
                except Exception as e:
                    err_msg = f"Error during live indexing for {rel_path}: {e}"
                    if self.callback:
                        self.callback(err_msg)
                    else:
                        print(err_msg)

    def stop(self):
        """Stops the worker thread."""
        self.running = False


class DirectoryWatcher:
    """Manages the watchdog Observer and scheduling handler events recursively."""
    
    def __init__(self, directory_path: str, manager: VectorStoreManager, callback: Optional[Callable[[str], None]] = None):
        self.directory_path = os.path.abspath(directory_path)
        self.manager = manager
        self.callback = callback
        self.handler = DebouncedReindexHandler(manager, self.directory_path, callback)
        self.observer = Observer()

    def start(self):
        """Starts monitoring the directory recursively."""
        self.observer.schedule(self.handler, path=self.directory_path, recursive=True)
        self.observer.start()

    def stop(self):
        """Stops monitoring and joins the watchdog thread."""
        self.observer.stop()
        self.handler.stop()
        self.observer.join()
