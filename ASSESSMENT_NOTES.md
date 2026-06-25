# Assessment Notes

## Assumptions

I assumed that the 2025 Harkalm acquisition brochure represented the authoritative business requirements and therefore used it as the basis for both the deterministic business rules and the prompt guidance.

I also assumed that property size was the most important objective criterion. Mandatory size requirements were therefore implemented in Python before invoking the language model, allowing categories that could not satisfy the acquisition criteria to be eliminated deterministically. Where structured size information was unavailable, the language model was permitted to consider any size information contained within the listing description.

Finally, I assumed that the input dataset would contain the minimum set of columns required for classification. The preprocessing stage validates these fields and raises an error if mandatory columns are missing, rather than attempting to infer missing information.

## Handling Ambiguous Listings

Commercial property listings frequently describe buildings as being suitable for a variety of alternative uses rather than a single purpose. My aim was to keep the classifier conservative, encouraging the model to remain grounded in the evidence contained within the listing rather than forcing a positive classification. Where no candidate category was sufficiently supported, the model returned **None**.

During development, I iteratively refined the prompt to encourage the model to assess a property's suitability for acquisition and future conversion, rather than simply identifying its current use. This improved several ambiguous cases, although some listings remain challenging. For example, former medical premises can still be under-classified despite appearing in Harkalm's acquisition criteria as suitable acquisition opportunities for future conversion.

## Future Improvements

Given more time, I would introduce automated tests (for example using `pytest`) for the preprocessing, business rules and response validation components. I would also evaluate the classifier against a labelled validation dataset to measure performance objectively.

From a modelling perspective, I would continue refining the reasoning process so that the model more consistently evaluates a property's current type against Harkalm's acquisition opportunities before assessing its suitability for conversion. Finally, I would investigate using a secondary LLM-assisted step to extract deterministic business rules from future acquisition documents, reducing the need for manual rule definition.