# Design decisions

Questions to answer in your own words before interviews. Keep the answers short and back them with your numbers.

1. Why does this project not use RAG? When would it be needed?
2. Why is `source_quote` mandatory? How is it checked, and how was the 0.85 fuzzy-match threshold chosen?
3. Why should the verifier be a different model from the generator?
4. Why are the answer options shuffled after generation?
5. How are documents split into sections, and why ~300-1,500 tokens? (ideally: results of the chunk-size experiment)
6. Which prompting strategy won, by how much, on which criteria? Is the difference meaningful given the sample size?
7. On which criteria does the judge agree with humans, and on which does it not? Why?
8. Small local model vs large hosted model: quality, cost, latency, privacy.
9. Did LoRA help the small model catch up? Where did the training data come from, and how was leakage into the
   evaluation set prevented?
10. Uploaded documents can contain prompt injection - how is it mitigated?
11. What would you change to serve 1,000 lecturers?
