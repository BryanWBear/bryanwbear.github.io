# Overview

This is the code for my blog. It uses Jeykll as a static site generator.

## Creating a new post from ipynb

1. First, run `jupyter nbconvert --to markdown post.ipynb` to get `post.md` and static files `post_files`
2. Move `post_files` to `assets/images` and `post.md` to `_posts`.
3. Rename `post.md` to `YYYY-MM-DD-post.md`, and add corresponding header into the file (can copy from the top of another post and modify).
4. Prepend `/assets/images/` to all of the image paths in `post.md`. TODO: this can be automated.
5. For latex support, we must add the following header to the front of the file, and change all $$ -> $$$$ (TODO: this can be automated): 

```
<head>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
<style>
.katex-display > .katex {
  display: inline-block;
  white-space: nowrap;
  max-width: 100%;
  overflow-x: scroll;
  text-align: initial;
}
.katex {
  font: normal 1.21em KaTeX_Main, Times New Roman, serif;
  line-height: 1.2;
  white-space: normal;
  text-indent: 0;
}
</style>
</head>
```
6. Run `bundle exec jekyll serve` to start the site locally.
7. If everything looks good, commit and push to Github. The new changes will be served automatically. 