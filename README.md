# Fun-With-Python3
Miscellaneous Python3 experiments intended for learning the language and for fun

All experiments will be in individual folders with their details mentioned in this file.

1. The Sorting folder.

The Sorting folder has a custom divide and conquer program similar to Merge Sort.
However, this custom program works in decremental logarithmic bases to reduce Actual Time (not time complexity/asymptotic time) of the program.
That is, for n inputs, the first iteration completes in O(n log n base(n/10)), followed by O(n log n base(n/100)), and so on.
As, log n base 100 < log n base 2, this actually takes lesser Actual time than merge sort. Although, it does not reduce asymptotic time complexity over Merge Sort, it does manage to complete earlier in real time by dividing the input arrays in more than two sub-arrays recursively and later merging them.

