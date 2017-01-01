# tf_basics

Set of examples of using tensorflow (http://tensorflow.org).<br>
Install tensorflow, then you can try this examples.


#xor.py:
 Neural network for calculating basic functions: xor, and, or, 1, 0.


sum.py:
 Neural network for calculating sum and product of two numbers.
 Usage example (ends up within 30 minutes): ./sum.py 3 2 0.1 1024 100


skip_gram.py:
 Word2vec (skip-gram) implementation.
 I trained word2vec on lib.ru data.
  1. Download archive (ftp://lib.ru/pub/moshkow/.library/).
  2. Unpack it.
  3. Glue all files into one (./preprocess_lib_ru.py > data/lib_ru).
  4. Train model (nohup ./skip_gram.py data/lib_ru dump.txt 0.001 0.1 64 65536 1000 100000 1000 1 5 1 10 мужчина женщина он она сказал сказала был была ему ей король королева дед бабка дедом бабкой кот кошка щенок собачка < /dev/null > 1.w2v 2> 1.w2v.err &). It takes several hours on i7 32GB GTX 1080. You can play with parameters and size of training data. Kill it when the test error becomes stable.
  5. Play with analogies_interactive.py . It trains matrix A and vector b - linear transform on embeddings to find analogies.
 My findings are follow. In the source tutorial (https://www.tensorflow.org/tutorials/word2vec/) they tell that we can simply add some vector (or calculate some kind of cos distance) to find analogies. I couldn't repeat this result, may be because of insufficient amount of training data.
 I trained Ax+b model over embeddings with a few training examples: having source vector x (embedding of source word) we apply Ax+b transform and get destination vector y (analogy for source word we need). We can find such (A, b) by minimizing Euclid distance between Y[i] and A * X[i] + b .
 Here I have some points.
  1. Such process gives us b close to 0 and A not so far from I. Matrix A is actually similar to some kind of rotation matrix. It's very interesting because in the source tutorial we have A=0 and b!=0.
  2. Having too few labeled data to train (A, b) we can't trust this results for sure. But several runs with different random seeds give pretty similar results (b=0 and A is a kind of rotation).

