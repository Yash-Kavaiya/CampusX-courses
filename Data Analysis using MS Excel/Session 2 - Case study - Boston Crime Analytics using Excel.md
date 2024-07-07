# Session 2 - Case study - Boston Crime Analytics using Excel

Excel is a powerful spreadsheet application developed by Microsoft. It is widely used for data analysis, financial modeling, and many other tasks that involve working with data. Excel includes numerous functions and commands to manipulate and analyze data. Two of the most commonly used functions for data lookup and reference are the `INDEX` and `MATCH` functions.

### INDEX Function

The `INDEX` function returns the value of a cell in a table based on the row and column numbers you specify.

**Syntax:**
```
INDEX(array, row_num, [column_num])
```

- `array`: The range of cells or an array constant.
- `row_num`: The row number in the array from which to return a value.
- `column_num` (optional): The column number in the array from which to return a value. If omitted, `INDEX` returns the value in the first column.

**Example:**
Suppose you have a table in the range `A1:C3`:
```
  A         B         C
1 Name      Age       Score
2 Alice     30        85
3 Bob       25        90
4 Charlie   35        95
```
If you want to find the score of Bob, you can use:
```
=INDEX(C2:C4, 2)
```
This formula returns `90`, which is the value in the second row of the range `C2:C4`.

### MATCH Function

The `MATCH` function searches for a specified value in a range and returns the relative position of that value within the range.

**Syntax:**
```
MATCH(lookup_value, lookup_array, [match_type])
```

- `lookup_value`: The value you want to search for.
- `lookup_array`: The range of cells being searched.
- `match_type` (optional): The number `-1`, `0`, or `1`. It determines how Excel matches the `lookup_value` with values in `lookup_array`.
  - `1` (default): Finds the largest value that is less than or equal to `lookup_value`. The `lookup_array` must be in ascending order.
  - `0`: Finds the first value that is exactly equal to `lookup_value`. The `lookup_array` can be in any order.
  - `-1`: Finds the smallest value that is greater than or equal to `lookup_value`. The `lookup_array` must be in descending order.

**Example:**
Using the same table as above, if you want to find the position of Bob in the range `A2:A4`, you can use:
```
=MATCH("Bob", A2:A4, 0)
```
This formula returns `2`, which is the position of "Bob" in the range `A2:A4`.

### Using INDEX and MATCH Together

The real power of these functions is realized when you use them together to perform lookups that are more flexible than what `VLOOKUP` can offer.

**Example:**
Suppose you want to find the score of a person whose name you specify in cell `E1`. You can use a combination of `INDEX` and `MATCH`:
```
=INDEX(C2:C4, MATCH(E1, A2:A4, 0))
```
If `E1` contains "Charlie", the formula will return `95`, which is Charlie's score.

### Practical Example with More Complex Data

Suppose you have a larger dataset in the range `A1:D5`:
```
  A         B         C       D
1 ID        Name      Age     Score
2 101       Alice     30      85
3 102       Bob       25      90
4 103       Charlie   35      95
5 104       David     28      88
```
You want to find the score of the person with ID `103`. Here's how you can do it using `INDEX` and `MATCH`:

1. Find the row number of ID `103`:
```
=MATCH(103, A2:A5, 0)
```
This returns `3` because ID `103` is in the third row of the range `A2:A5`.

2. Use the row number to get the corresponding score:
```
=INDEX(D2:D5, MATCH(103, A2:A5, 0))
```
This formula returns `95`, which is the score of the person with ID `103`.

By combining `INDEX` and `MATCH`, you can perform dynamic lookups that adjust based on the data and criteria you provide. This approach is more flexible and powerful compared to using `VLOOKUP`, especially when dealing with large datasets or when the lookup value is not in the first column.

The `VLOOKUP` function in Excel is used to search for a value in the first column of a table or range and return a value in the same row from a specified column. The "V" in `VLOOKUP` stands for "Vertical," indicating that it searches vertically down the first column of the table or range.

### Syntax

```
VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])
```

- `lookup_value`: The value you want to search for in the first column of the table.
- `table_array`: The range of cells that contains the data. The first column of this range will be searched.
- `col_index_num`: The column number (starting from 1 for the leftmost column of `table_array`) from which to retrieve the value.
- `range_lookup` (optional): A logical value that specifies whether you want an exact match (FALSE) or an approximate match (TRUE). If omitted, it defaults to TRUE.

### Example

Suppose you have the following table in the range `A1:D5`:

```
  A         B         C       D
1 ID        Name      Age     Score
2 101       Alice     30      85
3 102       Bob       25      90
4 103       Charlie   35      95
5 104       David     28      88
```

You want to find the score of the person with ID `103`. You can use `VLOOKUP` to do this:

```
=VLOOKUP(103, A2:D5, 4, FALSE)
```

- `lookup_value` is `103`.
- `table_array` is `A2:D5`.
- `col_index_num` is `4` (since "Score" is in the fourth column of `A2:D5`).
- `range_lookup` is `FALSE` (for an exact match).

This formula returns `95`, which is the score of the person with ID `103`.

### Detailed Explanation

1. **`lookup_value`**: This is the value you want to find in the first column of your `table_array`. In this case, it's `103`.

2. **`table_array`**: This is the range of cells that contains your data. The `lookup_value` is searched for in the first column of this range. Here, it's `A2:D5`.

3. **`col_index_num`**: This is the column number in the `table_array` from which to retrieve the value. The column number is relative to the `table_array`, not the worksheet. In this example, "Score" is the fourth column in the `A2:D5` range, so `col_index_num` is `4`.

4. **`range_lookup`**: This argument specifies whether you want an exact match or an approximate match:
   - `FALSE` means that you want an exact match.
   - `TRUE` means that you want an approximate match. If an exact match is not found, it will return the next largest value that is less than `lookup_value`.

### Important Considerations

- **Sorted Data for Approximate Match**: If you use `TRUE` (or omit the `range_lookup` argument), the first column of your `table_array` must be sorted in ascending order for `VLOOKUP` to return the correct result.
- **Exact Match**: If you use `FALSE` for `range_lookup`, `VLOOKUP` will search for an exact match. If it doesn't find an exact match, it will return `#N/A`.
- **Column Number**: The `col_index_num` must be within the range of `table_array`. If it's less than 1 or greater than the number of columns in `table_array`, `VLOOKUP` will return `#REF!`.

### Practical Example

Let's say you want to find the age of a person named "Bob" from the same table:

1. Find the row where the name "Bob" is located.
2. Return the value from the "Age" column in that row.

```
=VLOOKUP("Bob", A2:D5, 3, FALSE)
```

This formula returns `25`, which is the age of Bob.

### Limitations

- **Only Looks Right**: `VLOOKUP` can only search for the lookup value in the first column and return values from columns to the right. If you need to look left or perform more complex lookups, you may need to use `INDEX` and `MATCH` or `XLOOKUP` (available in newer versions of Excel).
- **Performance**: `VLOOKUP` can be slower than `INDEX` and `MATCH` for large datasets because it searches each row individually.
- **Exact Match Requirement**: If you need exact matches and your data is not sorted, `VLOOKUP` might not be the best choice.

Despite its limitations, `VLOOKUP` is a useful and widely used function in Excel for simple and quick lookups. For more complex scenarios, consider using `INDEX` and `MATCH` or `XLOOKUP`.

