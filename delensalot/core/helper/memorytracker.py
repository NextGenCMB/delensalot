import tracemalloc
import gc
import numpy as np

class MemoryTracker:
    def __init__(self):
        self.snapshots = {}
        tracemalloc.start()

    def take_snapshot(self, name="default"):
        """Store a memory snapshot with a given name."""
        self.snapshots[name] = tracemalloc.take_snapshot()

    def compare_snapshots(self, snapshot1, snapshot2, top_n=10):
        """Compare two snapshots and show the top memory differences."""
        if snapshot1 not in self.snapshots or snapshot2 not in self.snapshots:
            print(f"Snapshots '{snapshot1}' or '{snapshot2}' not found.")
            return

        stats = self.snapshots[snapshot2].compare_to(self.snapshots[snapshot1], "lineno")

        print(f"\n🔍 Memory difference between '{snapshot1}' and '{snapshot2}':")
        for stat in stats[:top_n]:  # Show top N changes
            print(stat)

    def print_top_allocations(self, top_n=10):
        """Print the largest memory allocations."""
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("traceback")

        print(f"\n📊 Top {top_n} memory-consuming allocations:")
        for stat in top_stats[:top_n]:
            print(stat)
            for line in stat.traceback.format():
                print(line)
            print("-" * 80)

    def find_large_numpy_arrays(self, top_n=10):
        """Find the largest NumPy arrays in memory."""
        all_objects = gc.get_objects()
        arrays = [obj for obj in all_objects if isinstance(obj, np.ndarray)]
        sorted_arrays = sorted(arrays, key=lambda x: x.nbytes, reverse=True)[:top_n]

        print("\n📦 Top NumPy arrays by memory usage:")
        for arr in sorted_arrays:
            print(f"  🔹 Shape: {arr.shape}, Dtype: {arr.dtype}, Size: {arr.nbytes / 1024 / 1024:.2f} MiB")

    def stop(self):
        """Stop tracemalloc tracking."""
        tracemalloc.stop()
