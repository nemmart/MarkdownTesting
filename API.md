## CGBN basic arithmetic
---
##### Set / Copy
`void set(cgbn_t &r, const cgbn_t &a)`

Copies the CGBN value of **_a_** into **_r_**. &nbsp; No return value.

##### Addition
`int32_t add(cgbn_t &r, const cgbn_t &a, const cgbn_t &b)`

Computes **_a + b_** and stores the result in **_r_**.  &nbsp; If the sum resulted in a carry out, then 1 is returned to all threads in the group, otherwise return 0. 

##### Subtraction

`int32_t sub(cgbn_t &r, const cgbn_t &a, const cgbn_t &b)`

Computes **_a - b_** and stores the result in **_r_**. &nbsp; If **_b>a_** then -1 is returned to all threads, otherwise return 0.

##### Multiplication

`void mul(cgbn_t &r, const cgbn_t &a, const cgbn_t &b)`

Computes the low half of the product of **_a \* b_**, upper half of product is discarded.  &nbsp; This is the CGBN equivalent of unsigned multiplication in C.

---

`void mul_high(cgbn_t &r, const cgbn_t &a, const cgbn_t &b)`

Computes the high half of the product of **_a \* b_**, lower half of product is discarded. 

---

`void sqr(cgbn_t &r, const cgbn_t &a)`

Computes the low half of the product of **_a \* a_**, upper half product is discarded.

---

`void sqr_high(cgbn_t &r, const cgbn_t &a)`

Computes the high half of the product of **_a \* a_**, lower half of product is discarded. 

