# Retrieval checks (story 4)

Hand checks of `src.retrieve.retrieve()` on the full index, run 2026-09-16 and re-run after the fingerprint fix and a clean rebuild. The re-run returned the same hits and distances.

**These verdicts are the implementing agent's judgement, not a human's and not a metric.** They are a sanity check that feeds story 6's gold set. They are not evaluation results.

## Setup

- Index: `all-MiniLM-L6-v2` embeddings of `review_text_normalized`, cosine space, built by `uv run python -m src.index`.
- Corpus: 53,872 reviews stored as 55,001 chunks. 928 reviews (1.72%) were longer than the 256-token window and were split.
- Build: 66.0 s on a clean `.chroma/` (Apple Silicon, MPS). A second run skipped embedding and exited 0 in 1.9 s wall time.
- On-disk size: 244 MB (`chroma.sqlite3` 144 MB plus a 100 MB HNSW segment).
- Retrieval: k = 6, brand and rating-band filters applied as Chroma `where` pre-filters. Snippets below are the matched chunk, cut short.
- Rating band `low` is 1–2★, `mid` is 3★, and `high` is 4–5★. The filter cannot select exactly 1★.
- Each query lists the command that reproduces it; the model runs offline once cached.
- Verdict scale: **relevant** means most hits address the question. **Partly** means the filters are right but under half the hits answer it. **Not** means the hits miss the intent.

## Summary

| # | Query | Filters | Verdict |
|---|-------|---------|---------|
| 1 | what do 1-star Pantene reviewers complain about | Pantene, low | partly |
| 2 | does Head & Shoulders actually get rid of dandruff | Head & Shoulders | relevant |
| 3 | Herbal Essences smell is too strong or gives headaches | Herbal Essences | **not** (miss) |
| 4 | shampoo made my hair fall out or caused hair loss | low | relevant |
| 5 | what do people like about the scent | high | partly |
| 6 | mixed feelings: works okay but too expensive for the size | mid | relevant |
| 7 | package arrived leaking or bottle broken in shipping | none | relevant |
| 8 | is it safe for color-treated hair | Pantene, high | relevant |

Result: 5 relevant, 2 partly, 1 not. Every hit in all 8 queries matched its filters (48 of 48).

## Queries

### 1. "what do 1-star Pantene reviewers complain about" (brand Pantene, band low)

```sh
uv run python -m src.retrieve "what do 1-star Pantene reviewers complain about" --brand Pantene --band low
```

| Brand | ★ | Distance | Snippet |
|---|---|---|---|
| Pantene | 1 | 0.428 | Will never purchase this product again!! The cheaper versions work so much better... HORRIBLE |
| Pantene | 1 | 0.465 | If I could give it zero stars I would. I will NEVER buy another Pantene product again. |
| Pantene | 1 | 0.483 | nothing good about this one... very disappointed with Pantene |
| Pantene | 2 | 0.490 | Pantene uses a lot of chemicals in their product which causes dry and brittle hair. |
| Pantene | 1 | 0.504 | This "new" formula leaves my hair dull and super staticky and extremely flyaway |
| Pantene | 1 | 0.505 | Pantene is my favorite not from this supplier way to much money |

**Partly.** The filters work, but the question asks for 1★ and the `low` band also includes 2★ (hit 4 is 2★). The band filter cannot express exactly 1★. Only hits 4 and 5 name a concrete complaint (dry and brittle hair, dull and staticky hair). The rest are angry reviews with no content that match on the word "Pantene". Hit 4 comes from a review split into 3 chunks (see the split-review check below).

Re-wording: "problems with this shampoo or conditioner", same filters.

```sh
uv run python -m src.retrieve "problems with this shampoo or conditioner" --brand Pantene --band low
```

| Brand | ★ | Distance | Snippet |
|---|---|---|---|
| Pantene | 1 | 0.205 | The wrong product was delivered twice... a two in one shampoo instead of conditioner |
| Pantene | 2 | 0.213 | I received 2 conditioners no shampoo |
| Pantene | 1 | 0.216 | Wrong item, I received the shampoo not the conditioner. |
| Pantene | 1 | 0.223 | I bought this and it came as conditioner not shampoo |
| Pantene | 2 | 0.224 | my hair had a sticky residue that made it look and feel like my hair was oily |
| Pantene | 1 | 0.225 | We got shampoo instead of conditioner |

The re-wording is no better. 5 of 6 hits are wrong-item deliveries, which match "shampoo or conditioner" literally, and only hit 5 (sticky, oily residue) is a product complaint. This matters because this is the headline demo question. A meta question ("what do they complain about") has no single semantic target, so top-6 similarity returns one cluster of reviews instead of a spread of themes. Story 5 should expect thin answers here.

### 2. "does Head & Shoulders actually get rid of dandruff" (brand Head & Shoulders)

```sh
uv run python -m src.retrieve "does Head & Shoulders actually get rid of dandruff" --brand "Head & Shoulders"
```

| Brand | ★ | Distance | Snippet |
|---|---|---|---|
| Head & Shoulders | 4 | 0.143 | Head & Shoulders has helped me in past with serious dandruff issues. |
| Head & Shoulders | 5 | 0.148 | works great. I have no had dandruff since using Head and Shoulders. |
| Head & Shoulders | 3 | 0.149 | I'm not sure that this is supposed to get rid of dandruff. |
| Head & Shoulders | 5 | 0.166 | did someone say dandruff? not anymore with head and shoulders |
| Head & Shoulders | 5 | 0.176 | if used daily or even weekly it will eliminate dandruff |
| Head & Shoulders | 5 | 0.178 | the only head and shoulders that actually helps with my boyfriends dandruff |

**Relevant.** All 6 hits speak to dandruff efficacy, and one is a doubtful 3★ review. The hits are short, so they give little detail.

### 3. "Herbal Essences smell is too strong or gives headaches" (brand Herbal Essences)

```sh
uv run python -m src.retrieve "Herbal Essences smell is too strong or gives headaches" --brand "Herbal Essences"
```

| Brand | ★ | Distance | Snippet |
|---|---|---|---|
| Herbal Essences | 5 | 0.284 | Herbal essences body smell really good will but again reminds me of the shampoo |
| Herbal Essences | 1 | 0.284 | Worst smelling herbal essences ever. Blehh. |
| Herbal Essences | 5 | 0.295 | Best smelling Herbal Essence in my opinion. |
| Herbal Essences | 4 | 0.299 | The smell is faint and doesn't last through the day. |
| Herbal Essences | 5 | 0.300 | The one thing that sets Herbal Essence apart from others is the smell. |
| Herbal Essences | 5 | 0.306 | Best smelling herbal essence out there. |

**Not (a miss).** The hits are about "Herbal Essences" plus "smell". Four of them praise the scent, one says it is too faint, and none mention a strong scent or headaches. The corpus does contain the answer, as a re-wording without the brand name shows (same brand filter):

```sh
uv run python -m src.retrieve "the fragrance is overpowering and gave me a headache" --brand "Herbal Essences"
```

| Brand | ★ | Distance | Snippet |
|---|---|---|---|
| Herbal Essences | 1 | 0.209 | The scent was so strong it gave me a headache. |
| Herbal Essences | 2 | 0.209 | i really don't like the scent, its too strong and gives me a headache. |
| Herbal Essences | 3 | 0.228 | The scent is overwhelming. I have headaches when I used this shampoo and conditioner. |
| Herbal Essences | 1 | 0.243 | The fragrance is so awful I won't be using it again. |
| Herbal Essences | 2 | 0.259 | Did not like the scent! It is to strong and overwhelming... Gave me a headache. |
| Herbal Essences | 3 | 0.267 | The scent on this was horribly overpowering. |

5 of 6 re-worded hits are on point. Hit 4 is a dislike of the scent without saying it is strong. The likely cause is that the brand name in the query pulls toward short reviews that repeat the brand name, which the brand filter already handles. Candidate fix for later stories: strip the filtered brand name from the query before embedding. It was not applied here.

### 4. "shampoo made my hair fall out or caused hair loss" (band low)

```sh
uv run python -m src.retrieve "shampoo made my hair fall out or caused hair loss" --band low
```

| Brand | ★ | Distance | Snippet |
|---|---|---|---|
| Herbal Essences | 1 | 0.212 | I started to loss hair after a week of using this shampoo |
| Pantene | 2 | 0.214 | after using this shampoo, i got serious hair loss. |
| Herbal Essences | 1 | 0.237 | Makes your hair fall out like crazy... MY HAIR VISIBLY THINNED. |
| Herbal Essences | 1 | 0.241 | My hair feels very dried out while using this shampoo |
| Herbal Essences | 1 | 0.248 | quite a few other customers also experienced hair loss |
| Herbal Essences | 1 | 0.250 | This shampoo made my hair fall out more in the shower and my scalp was really dry |

**Relevant.** 5 of 6 hits describe hair loss. Hit 4 is about dryness, not hair loss. Five of the six are Herbal Essences, so an unfiltered question can lean toward one brand. That is a retrieval observation, not a measured brand comparison.

### 5. "what do people like about the scent" (band high)

```sh
uv run python -m src.retrieve "what do people like about the scent" --band high
```

| Brand | ★ | Distance | Snippet |
|---|---|---|---|
| Herbal Essences | 5 | 0.170 | The scent is pleasant and not overbearing |
| Pantene | 5 | 0.184 | The scent is just wonderful |
| Pantene | 5 | 0.204 | I really like the scent. |
| Head & Shoulders | 4 | 0.210 | Scent isn't bad. It works well though and lots of it! |
| Pantene | 5 | 0.211 | The product's scent is nice . |
| Pantene | 5 | 0.211 | Great scent - my wife and daughters love it |

**Partly.** All 6 hits praise the scent, but none say *what* people like about it (a note, a strength, how long it lasts), so under half answer the question. Short generic reviews rank highest. This is the same pattern as query 1: very short reviews sit close to short questions.

### 6. "mixed feelings: works okay but too expensive for the size" (band mid)

```sh
uv run python -m src.retrieve "mixed feelings: works okay but too expensive for the size" --band mid
```

| Brand | ★ | Distance | Snippet |
|---|---|---|---|
| Pantene | 3 | 0.452 | Love this product, but deceiving small size. |
| Pantene | 3 | 0.453 | Cost too much for such a small bottle |
| Herbal Essences | 3 | 0.486 | The products are okay, but they are over priced |
| Pantene | 3 | 0.506 | for the price I could not go wrong size is quite large and will last a long time |
| Pantene | 3 | 0.509 | A brand you can buy anywhere, better price point. It's ok. |
| Pantene | 3 | 0.514 | Good product, teeny tiny tube size. For the same price or less, you can buy the full size in stores |

**Relevant.** Hits 1, 2, 3, and 6 match. Hit 5 is loosely about price. Hit 4 says the opposite (good price, large size): the topic matches but the stance does not, a known weakness of embedding similarity.

### 7. "package arrived leaking or bottle broken in shipping" (no filters)

```sh
uv run python -m src.retrieve "package arrived leaking or bottle broken in shipping"
```

| Brand | ★ | Distance | Snippet |
|---|---|---|---|
| Pantene | 1 | 0.254 | Bottle arrived cracked on bottom liquid leaking out |
| Pantene | 1 | 0.256 | bottle was received busted |
| Head & Shoulders | 3 | 0.262 | one of the shampoo bottles was popped open and leaked all over the box |
| Pantene | 1 | 0.266 | Half bottles were waisted, package arrived leaked. |
| Pantene | 1 | 0.270 | ALL the bottle were broken at the cap and leaking. |
| Head & Shoulders | 5 | 0.271 | Packaging came damaging & leaking. Washed off the bottles. Product works great. |

**Relevant.** All 6 hits describe damaged or leaking deliveries. With no rating filter, a 3★ and a 5★ review show up too.

### 8. "is it safe for color-treated hair" (brand Pantene, band high)

```sh
uv run python -m src.retrieve "is it safe for color-treated hair" --brand Pantene --band high
```

| Brand | ★ | Distance | Snippet |
|---|---|---|---|
| Pantene | 5 | 0.231 | Excellent for color treated hair |
| Pantene | 5 | 0.247 | I've used this product for quite some time on my color treated hair |
| Pantene | 5 | 0.260 | Perfect product to keep color treated hair healthy. |
| Pantene | 4 | 0.270 | Safe on colored hair, really nice scent |
| Pantene | 5 | 0.273 | I like that it is safe for color treated hair. My blond streaks stay bright |
| Pantene | 5 | 0.295 | This is the best conditioner for color treated hair EVER! |

**Relevant.** All 6 hits address color-treated hair. The high band makes the result one-sided by design.

## Split-review check

928 reviews were longer than the window and were split. This check confirms that a hit from one of them returns the matching chunk plus the full original text. Query 1's hit 4 (`c95b6ec0e2da6a93`, Pantene 2★) is stored as 3 chunks (`n_chunks = 3`). The hit is chunk 2, 681 characters long, starting at character 1,969 of the 2,673-character review.

```sh
uv run python -m src.retrieve "what do 1-star Pantene reviewers complain about" --brand Pantene --band low
```

- `chunk_text` starts: `/> Even though I had some pretty good reasons as to why Pantene was a very good product, its disadvantages are truly important to address.`
- `review_text` starts: `I began using the product Pantene several years ago.  My mother and I have been trying to find products...`

The hit appears once, not once per chunk. Its `chunk_text` is the matching late chunk, and its `review_text` is the whole original review with its double spaces kept. The chunk opens with `/>`, left over from `<br />` markup in the source, because the chunk boundary falls inside that markup.

## Observations for stories 5 and 6

- Filters are reliable: 48 of 48 hits matched brand and band.
- Distances do not track relevance across queries. The query 3 miss (0.284 to 0.306) scored closer than the partly-relevant query 1 (0.428 to 0.505). A fixed distance cutoff would drop query 1 and keep query 3, so story 5 should not use one without measuring it first.
- Short, generic reviews rank high on short or broad questions (queries 1 and 5, both partly). A "what do people complain about" answer needs varied hits, which plain top-6 similarity does not provide.
- A brand name repeated in the query can crowd out on-topic reviews (query 3).
