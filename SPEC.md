# Mila

## Types

| Name    | Notation  | Description           |
| ------- | --------  | -----------           |
| array   | `array [istart .. iend] of type` | declares an array with length (istart - iend) where istart is the first and iend the last index |
| integer | `integer` | 64-bit integer        |
| double  | `double`  | 64-bit floating-point |

## Expressions

### Binary infix operators

| Literal  | Description           |
| -------- | -----------           |
| `+`      | addition              |
| `-`      | subtraction           |
| `*`      | multiplication        |
| `div`    | division              |
| `mod`    | modulo                |
| `:=`     | becomes               |
| `>=`     | greater than or equal |
| `<=`     | less than or equal    |
| `=`      | equal                 |
| `<>`     | not equal             |
| `>`      | greater than          |
| `<`      | less than             |
| `and`    | logical and           |
| `or`     | logical or            |
| `to`     | ascending range       |
| `downto` | descending range      |

### Postfix operators

| Literal | Description     |
| ------- | -----------     |
| `[]`    | array subscript |
| `()`    | function call   |

## Statements

Statements need to be delimited if immediately followed by a new statement.

Delimited expressions become expression statements.

### Keywords

| Keyword    | Description               |
| -------    | -----------               |
| `program`  | named module              |
| `var`      | variable declaration      |
| `const`    | constant definition       |
| `procedure | procedure declaration     |
| `function` | function declaration      |
| `forward`  | forward declaration       |
| `exit`     | exits function or program |

### Control flow

| Control                                         | Description        |
| -------                                         | -----------        | 
| `begin ([[statement]];)+ end`                   | compound statement |
| `for [[range assignment]] do [[statement]]`     | for loop           |
| `if [[boolean expression]] then [[statement]]` <br> `else [[statement]]` | condition branching |
| `while [[boolean expression]] do [[statement]]` | while loop         |

## Built-in functions

| Name        | Description                                            |
| ----        | -----------                                            |
| `dec`       | Decrement integer.                                     |
| `inc`       | Increment integer.                                     |
| `readln`    | Read to variable from a line of standard input.        |
| `write`     | Write variable to standard output.                     |
| `writeln`   | Write variable to standard output and append new line. |

## Miscellaneous

| Literal | Description         |
| ------- | -----------         |
| `;`     | statement delimiter |
| `:`     | type denotation     |
| `(`     | opening parenthesis |
| `)`     | closing parenthesis |
| `,`     | comma  				  |
