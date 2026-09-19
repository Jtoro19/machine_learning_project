# Stratified Subsampling Specification

## Purpose

Give the Gaussian Process classifier, spectral clustering, and t-SNE notebooks a single, deterministic way to reduce a cleaned partition to a workable size while keeping every `attack_cat` class visible. This capability restates the relevant `AGENTS.md` data rules directly: Gaussian Process, spectral clustering, and t-SNE run on a stratified subsample of at most 10,000 rows, and the seed for every subsample is 42.

The subsample floor deliberately breaks strict proportionality for minority classes. This is a disclosed, mandatory trade-off, not a defect: every consumer of the subsample MUST be able to see, per class, whether the floor altered that class's allocation.

## Requirements

### Requirement: Deterministic seeded stratified subsample bounded at 10,000 rows

The system MUST provide a stratified subsampling function that selects at most 10,000 rows from an input partition, stratified by `attack_cat`, using a fixed random seed of 42 by default. Given the same input partition, the same parameters, and the same seed, repeated calls MUST return identical row selections.

#### Scenario: Repeated calls with the same seed return identical rows

- GIVEN a cleaned training partition
- WHEN the stratified subsample function is called twice with identical parameters, including the default seed
- THEN both calls MUST return the exact same set of row indices

#### Scenario: The output never exceeds the row cap

- GIVEN a cleaned training partition larger than 10,000 rows
- WHEN the stratified subsample function is called with the default maximum of 10,000 rows
- THEN the returned row count MUST be less than or equal to 10,000

### Requirement: Minimum-per-class floor with disclosed distortion

The subsample function MUST accept a minimum-per-class floor parameter, defaulting to 50, that guarantees every `attack_cat` class present in the input receives at least that many rows when the input contains that many for the class. Setting the floor to 0 MUST reproduce pure proportional stratified allocation, with no class receiving special treatment. The function MUST return, for every class, the realized row count actually selected and a boolean flag indicating whether the floor altered that class's allocation relative to pure proportional allocation.

#### Scenario: Every class is retained at the default floor

- GIVEN a cleaned training partition containing all ten `attack_cat` classes, including a class whose pure proportional share of a 10,000-row subsample would fall below 50 rows
- WHEN the stratified subsample function runs with the default floor of 50
- THEN every one of the ten classes MUST appear in the output
- AND the class whose proportional share was below 50 MUST receive at least 50 rows, provided the input contains at least 50 rows for that class

#### Scenario: A floor of zero reproduces pure proportional allocation

- GIVEN a cleaned training partition containing multiple `attack_cat` classes
- WHEN the stratified subsample function runs with `floor=0`
- THEN every class's realized row count MUST match its pure proportional share of the requested row cap, within the rounding tolerance inherent to allocating whole rows
- AND every class's floor-applied flag MUST be `False`

#### Scenario: Realized counts and floor-applied flags are always returned

- GIVEN any successful call to the stratified subsample function
- WHEN its return value is inspected
- THEN it MUST include, for every class present in the input, the realized row count selected for that class
- AND it MUST include, for every class present in the input, a boolean flag indicating whether the floor changed that class's allocation from its pure proportional share

### Requirement: A class smaller than the floor yields all of its available rows

When a class's total row count in the input partition is smaller than the configured floor, the subsample function MUST select every available row for that class rather than raising an error or silently omitting the class. The floor-applied flag for that class MUST still indicate the class received different treatment than pure proportional allocation would have produced, because it received all its rows rather than a proportionally computed subset.

#### Scenario: A class with fewer rows than the floor contributes all of its rows

- GIVEN a cleaned training partition where a class's total row count is smaller than the configured floor
- WHEN the stratified subsample function runs
- THEN every row belonging to that class MUST be included in the output
- AND the function MUST NOT raise an error because the class could not reach the nominal floor value

### Requirement: An input at or below the row cap is returned in full, unsampled

When the input partition's row count is already less than or equal to the requested maximum row count, the subsample function MUST return the entire input partition without dropping any row, and MUST report every class's floor-applied flag as `False`, because no reduction was necessary or performed.

#### Scenario: An input already within the row cap is returned unchanged

- GIVEN a cleaned partition whose total row count is less than or equal to the requested maximum row count
- WHEN the stratified subsample function runs
- THEN the returned row count MUST equal the input row count
- AND every row of the input MUST be present in the output
- AND every class's floor-applied flag MUST be `False`

### Requirement: Deterministic ordering as a precondition for reproducibility

The subsample function's input MUST have a deterministic row order or index before sampling, so that seed-42 reproducibility is not silently broken by a non-deterministic upstream row order.

#### Scenario: Identical results despite differently-ordered but content-equal input

- GIVEN two data frames containing the same rows as a cleaned training partition, but constructed with different underlying row orderings before their index is reset
- WHEN the stratified subsample function is called on each with an index reset to a deterministic order beforehand, using identical parameters and seed
- THEN both calls MUST select the same underlying rows
