[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/DClpi-Xf)

## 4/11/2025 log

the algorithm is adapting, but not well, and kinda improves and gets worse in waves.

We are still getting, Z as the tree result (presumably) every time.

Ran into issues implementing booleans, need to ask Nic about in class.

## Notes from earlier stash

I had a chance to talk to Nic after class about the functions we include in the GA's toolbox. These should be added around line 70. Im not entirely sure what constitutes the boolean functions group, but the one function he recommended we include from 'exec' is an if statement. (other functions in 'exec' were loop variants).

Problem Description:
"Grade Given 5 integers, the first four represent the lower numeric thresholds for achieving an A, B, C, and D, and will be distinct and in descending order. The fifth represents the student’s numeric grade. The program must print Student has a X grade., where X is A, B, C, D, or F depending on the thresholds and the numeric grade.

Specifics

| Name | Inputs | Outputs | Train | Test |
|------|------------|-------|-------|------|
|Grade | 5 integers in [0, 100] | printed string | 200 | 2000 |

Data types uses:

- exec --> if()
- integer --> mult(), add(), etc...
- boolean --> not(), and(), or(), etc...
- string --> I think these where concatenation commands
- print --> print()

Constants and strings provided were: "Student has a ”, “ grade.”, “A”, “B”, “C”, “D”, “F”, integer ERC
