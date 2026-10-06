# The project book

`Worker_Safety_PPE_Monitoring_System_Book.pdf` is the book this project is built from.

- **Parts I–III** explain the ideas. They stay as first written.
- **Part IV** is the phase-by-phase build guide. Each phase that has been built ends with an
  **As Built** section: what was built, what changed from the plan and why, the measured
  results, and the lessons.
- **Appendix F** is the dated build log. **Appendix G** is the guide to using the system (the short
  form of `docs/USER_GUIDE.md`).

The book was updated at every milestone: a phase reviewed, a new measurement, a change of plan.
The sixth edition (29 Sep 2026) is the project complete.

## Rebuilding it

```bash
pip install reportlab==4.4.10            # once
python docs/book/make_figures.py         # only when a diagram or chart changed
python docs/book/build_book.py           # writes docs/book/Worker_Safety_PPE_Monitoring_System_Book.pdf
```

- **Build notes:** `AS_BUILT_17` … `AS_BUILT_23` in `build_book.py`.
- **Build log:** Appendix F, near the end of `build_book.py`.
- **Edition line:** `EDITION`, near the top of `build_book.py`.
- **Charts:** `make_figures.py`. Every number in them comes from `docs/phase*_results.md` or
  `docs/real_clips.md`.
