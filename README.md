# MatrixProduct
Project1

##矩阵乘法优化测试报告
* 本次报告共分为三个部分。第一部分是对于矩阵乘法加速的各个算法的尝试与测试结果。第二部分是实际的应用程序，即从第一部分的诸多算法中选择效率最好的算法来编写用户可实际使用的应用程序。
* 第一部分的内容主要是向老师呈现我整个优化过程中的思路历程以及所用方法的效果。关于程序在应用上的优缺点以及适用范围将在第二部分详细阐述。
* 第三部分是报告总结。

## 一：测试部分：

* 注意：由于在优化程度较低的情况下程序运行速度较慢，故若一开始就将矩阵大小设置为10000*10000则运行时间会相当长久。故在测试阶段我采用的是1000 *1000的由随机数生成的矩阵。待选出优化效率较高的程序后则会测试10000 *10000的运行时间。

* 首先定义矩阵结构matrix：

  ```c++
  struct matrix
  {
  	int row;
  	int col;
  	float** data;
  };
  ```

  其中data指的是矩阵当中的元素。

* 现在我们不做任何优化，直接用最简单的三重循环模拟行乘列的矩阵乘法，测试它所需要的时间：

  ![Screenshot 2020-11-15 193445](E:\code\MatrixProduct\picture\Screenshot 2020-11-15 193445.png)

这里我们将这种方法命名为normal，用时接近6秒。
* 然后使用O3优化进行编译，再测试时间：

  ![Screenshot 2020-11-15 193756](E:\code\MatrixProduct\picture\Screenshot 2020-11-15 193756.png)
  
* 可以看到速度有了明显的提升。注意由于矩阵是由随机数生成的，所以每次运行最终得到相同位置矩阵元素的结果并不相同。之后我会将各个方法同时运行以验证结果的正确性。

* 注意：接下来的方法时间测试时均已进行O3优化。

* 最开始我尝试了最为简单的分块算法，代码如下：

  ```c++
  void block1(matrix& m, matrix& n, matrix& g)
  {
  	for (int i = 0; i < g.row; i++)
  	{
  		for(int j = 0; j < g.col; j++)
  		{
  			g.data[i][j] = 0;
  			for (int l = 0; l < m.col; l += 8)
  			{
  				g.data[i][j] +=
  					m.data[i][l] * n.data[l][j] +
  					m.data[i][l + 1] * n.data[l + 1][j] +
  					m.data[i][l + 2] * n.data[l + 2][j] +
  					m.data[i][l + 3] * n.data[l + 3][j] +
  					m.data[i][l + 4] * n.data[l + 4][j] +
  					m.data[i][l + 5] * n.data[l + 5][j] +
  					m.data[i][l + 6] * n.data[l + 6][j] +
  					m.data[i][l + 7] * n.data[l + 7][j];
  			}
  		}
  	}
  }
  ```

  这是最为简单直白的分块，仅仅只是在第一种方法上加以一点点的改动。运行结果如下：

  ![Screenshot 2020-11-15 194632](E:\code\MatrixProduct\picture\Screenshot 2020-11-15 194632.png)
  
* 提速不到百分之20，效果不大。

* 接着我考虑使用一种稍微复杂一些的分块方法，即每次处理四行和四列，相应代码如下

```c++
void block2(matrix& m, matrix& n, matrix& g)
{
	for (int i = 0; i < g.row - 4; i += 4)
	{
		for (int j = 0; j < g.col - 4; j += 4)
		{
			g.data[i][j] = 0;
			g.data[i + 1][j] = 0;
			g.data[i + 2][j] = 0;
			g.data[i + 3][j] = 0;
			g.data[i][j + 1] = 0;
			g.data[i + 1][j + 1] = 0;
			g.data[i + 2][j + 1] = 0;
			g.data[i + 3][j + 1] = 0;
			g.data[i][j + 2] = 0;
			g.data[i + 1][j + 2] = 0;
			g.data[i + 2][j + 2] = 0;
			g.data[i + 3][j + 2] = 0;
			g.data[i][j + 3] = 0;
			g.data[i + 1][j + 3] = 0;
			g.data[i + 2][j + 3] = 0;
			g.data[i + 3][j + 3] = 0;
			for (int l = 0; l < m.col; l++)
			{
				g.data[i][j] += m.data[i][l] * n.data[l][j];
				g.data[i + 1][j] += m.data[i + 1][l] * n.data[l][j];
				g.data[i + 2][j] += m.data[i + 2][l] * n.data[l][j];
				g.data[i + 3][j] += m.data[i + 3][l] * n.data[l][j];
				g.data[i][j + 1] += m.data[i][l] * n.data[l][j + 1];
				g.data[i + 1][j + 1] += m.data[i + 1][l] * n.data[l][j + 1];
				g.data[i + 2][j + 1] += m.data[i + 2][l] * n.data[l][j + 1];
				g.data[i + 3][j + 1] += m.data[i + 3][l] * n.data[l][j + 1];
				g.data[i][j + 2] += m.data[i][l] * n.data[l][j + 2];
				g.data[i + 1][j + 2] += m.data[i + 1][l] * n.data[l][j + 2];
				g.data[i + 2][j + 2] += m.data[i + 2][l] * n.data[l][j + 2];
				g.data[i + 3][j + 2] += m.data[i + 3][l] * n.data[l][j + 2];
				g.data[i][j + 3] += m.data[i][l] * n.data[l][j + 3];
				g.data[i + 1][j + 3] += m.data[i + 1][l] * n.data[l][j + 3];
				g.data[i + 2][j + 3] += m.data[i + 2][l] * n.data[l][j + 3];
				g.data[i + 3][j + 3] += m.data[i + 3][l] * n.data[l][j + 3];
			}
		}
	}
}
```

称该方法为block2，测试时间为：

![Screenshot 2020-11-15 200115](E:\code\MatrixProduct\picture\Screenshot 2020-11-15 200115.png)

优化超过百分之30，当然这离我们的目标还远远不够，但至少这样的分块是有作用的。之后我会将这种分块方法与其他优化方法相结合以进一步提高效率。

* 接下来的两个方法我的主要思路是尽量使得访存连续以提高计算效率。由于我的矩阵是按行储存，故当指针移动时只能按位读取一行上的数据，读取列上数据则无法简单的通过移动指针来进行。所以我在以上面分块方式的基础上（即每次处理四行和四列的乘法），将每次处理的四列数据先储存在一个只有四行的矩阵中，通过这个小矩阵来使得它的访存变为连续。具体实现如下：

  ```c++
  void senior_block(matrix& m, float** n, matrix& g, int row, int col)
  {
  	register float t0(0), t1(0), t2(0), t3(0), t4(0), t5(0), t6(0), t7(0),
  		t8(0), t9(0), t10(0), t11(0), t12(0), t13(0), t14(0), t15(0);
  	float *a0(m.data[row]), * a1(m.data[row+1]), * a2(m.data[row+2]), * a3(m.data[row+3]),
  		* b0(n[0]), * b1(n[1]), * b2(n[2]), * b3(n[3]), * end = a0 + m.col;
  	do {
  		t0 += *(a0) * *(b0);
  		t1 += *(a1) * *(b0);
  		t2 += *(a2) * *(b0);
  		t3 += *(a3) * *(b0++);
  		t4 += *(a0) * *(b1);
  		t5 += *(a1) * *(b1);
  		t6 += *(a2) * *(b1);
  		t7 += *(a3) * *(b1++);
  		t8 += *(a0) * *(b2);
  		t9 += *(a1) * *(b2);
  		t10 += *(a2) * *(b2);
  		t11 += *(a3) * *(b2++);
  		t12 += *(a0++) * *(b3);
  		t13 += *(a1++) * *(b3);
  		t14 += *(a2++) * *(b3);
  		t15 += *(a3++) * *(b3++);
  	} while (a0 != end);
  	g.data[row][col] = t0;
  	g.data[row + 1][col] = t1;
  	g.data[row + 2][col] = t2;
  	g.data[row + 3][col] = t3;
  	g.data[row][col + 1] = t4;
  	g.data[row+1][col + 1] = t5;
  	g.data[row+2][col + 1] = t6;
  	g.data[row+3][col + 1] = t7;
  	g.data[row][col + 2] = t8;
  	g.data[row+1][col + 2] = t9;
  	g.data[row+2][col + 2] = t10;
  	g.data[row+3][col + 2] = t11;
  	g.data[row][col + 3] = t12;
  	g.data[row+1][col + 3] = t13;
  	g.data[row+2][col + 3] = t14;
  	g.data[row+3][col + 3] = t15;
  }
  matrix result(matrix& A, matrix& B)
  {
  	int A_row = A.row;
  	int B_col = B.col;
  	int B_row = B.row;
  	matrix tp;
  	tp.data = matrix_2d(A_row, B_col);
  	float** tr = matrix_2d(4, B_row);
  	int i = 0;
  	int j = 0;
  	do {
  		j = 0;
  		do {
  			tr[0][j] = B.data[j][i];
  			tr[1][j] = B.data[j][i+1];
  			tr[2][j] = B.data[j][i+2];
  			tr[3][j] = B.data[j][i+3];
  		} while ((++j) < B.row);
  		j = 0;
  		do {
  			senior_block(A, tr, tp, j, i);
  			j += 4;
  		} while (j < B_row);
  		i += 4;
  	} while (i< B_col);
  	return tp;
  }
  ```

  以上共有两个函数，第一个函数用于计算四行与四列乘积，得到4 * 4的部分结果。第二行是一个“packing”过程，即将数据“打包”到小矩阵里使得访存连续。这种“packing”方法我是借鉴了CSDN上一位博主的做法。运行结果如下：

  ![Screenshot 2020-11-15 201603](E:\code\MatrixProduct\picture\Screenshot 2020-11-15 201603.png)

这时候已经提升约百分之75.
* 在写完这个代码后，继续思考了有关访存连续的问题，觉得上述使用的所谓“packing”方法还是太过繁琐，我认为如果在计算A*B之前先将矩阵B转置，则乘法就从行乘列变为了行乘行，那么访存就自然变得连续了。我认为这样的方法会有更好的优化效果。下为具体代码：

  ```c++
  void transport(matrix& g)
  {
  	matrix tmp;
  	setmatrix(g.row, g.col, tmp);
  	for (int i = 0; i < g.row; i++)
  	{
  		for (int j = 0; j < g.col; j++)
  		{
  			tmp.data[i][j] = g.data[i][j];
  		}
  	}
  void senior_block2(matrix& m, matrix& n, matrix& g, int i, int j)
  {
  	register float t0(0), t1(0), t2(0), t3(0), t4(0), t5(0), t6(0), t7(0),
  		t8(0), t9(0), t10(0), t11(0), t12(0), t13(0), t14(0), t15(0);
  	float* a0(m.data[i]), * a1(m.data[i + 1]), * a2(m.data[i + 2]), * a3(m.data[i + 3]),
  		* b0(n.data[j]), * b1(n.data[j + 1]), * b2(n.data[j + 2]), * b3(n.data[j + 3]), * end = a0 + m.col;
  	do {
  		t0 += *(a0) * *(b0);
  		t1 += *(a1) * *(b0);
  		t2 += *(a2) * *(b0);
  		t3 += *(a3) * *(b0++);
  		t4 += *(a0) * *(b1);
  		t5 += *(a1) * *(b1);
  		t6 += *(a2) * *(b1);
  		t7 += *(a3) * *(b1++);
  		t8 += *(a0) * *(b2);
  		t9 += *(a1) * *(b2);
  		t10 += *(a2) * *(b2);
  		t11 += *(a3) * *(b2++);
  		t12 += *(a0++) * *(b3);
  		t13 += *(a1++) * *(b3);
  		t14 += *(a2++) * *(b3);
  		t15 += *(a3++) * *(b3++);
  	} while (a0 != end);
  	g.data[i][j] = t0;
  	g.data[i + 1][j] = t1;
  	g.data[i + 2][j] = t2;
  	g.data[i + 3][j] = t3;
  	g.data[i][j + 1] = t4;
  	g.data[i + 1][j + 1] = t5;
  	g.data[i + 2][j + 1] = t6;
  	g.data[i + 3][j + 1] = t7;
  	g.data[i][j + 2] = t8;
  	g.data[i + 1][j + 2] = t9;
  	g.data[i + 2][j + 2] = t10;
  	g.data[i + 3][j + 2] = t11;
  	g.data[i][j + 3] = t12;
  	g.data[i + 1][j + 3] = t13;
  	g.data[i + 2][j + 3] = t14;
  	g.data[i + 3][j + 3] = t15;
  }
  ```

  上述代码共两个函数，第一个是用来转置的函数，第二个是计算函数。测试结果如下：

  ![Screenshot 2020-11-15 202354](E:\code\MatrixProduct\picture\Screenshot 2020-11-15 202354.png)

遗憾的是，我的方法与上面方法效率几乎没有任何区别。
* 目前看来我在访存上的优化已经遇到瓶颈，于是我想尝试老师上课提到的SSE指令集加速。我将上述senior__block2方法用SSE指令集重写了一遍，代码如下：

  ```c++
  void sse_senior_block(matrix& m, matrix& n, matrix& g, int i, int j)
  {
  			float* a0(m.data[i]), * a1(m.data[i + 1]), * a2(m.data[i + 2]), * a3(m.data[i + 3]),
  				* b0(n.data[j]), * b1(n.data[j+1]), * b2(n.data[j+2]), * b3(n.data[j+3]), * end = a0 + m.col;
  			__m128 m0, m1, m2, m3, n0, n1, n2, n3;
  			__m128 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
  			t0 = t1 = t2 = t3 = t4 = t5 = t6 = t7 = t8 = t9 = t10 = t11 = t12 = t13 = t14 = t15 = _mm_set1_ps(0);
  			do {
  				m0 = _mm_load_ps(a0); m1 = _mm_load_ps(a1); m2 = _mm_load_ps(a2); m3 = _mm_load_ps(a3);
  				n0 = _mm_set1_ps(*(b0)++); n1 = _mm_set1_ps(*(b1)++); n2 = _mm_set1_ps(*(b2)++); n3 = _mm_set1_ps(*(b3)++);
  				t0 = _mm_add_ps(t0, _mm_mul_ps(m0, n0));
  				t1 =_mm_add_ps(t1, _mm_mul_ps(m1, n0));
  				t2 = _mm_add_ps(t2, _mm_mul_ps(m2, n0));
  				t3 = _mm_add_ps(t3, _mm_mul_ps(m3, n0));
  				t4 = _mm_add_ps(t4, _mm_mul_ps(m0, n1));
  				t5 = _mm_add_ps(t5, _mm_mul_ps(m1, n1));
  				t6 = _mm_add_ps(t6, _mm_mul_ps(m2, n1));
  				t7 = _mm_add_ps(t7, _mm_mul_ps(m3, n1));
  				t8 = _mm_add_ps(t8, _mm_mul_ps(m0, n2));
  				t9 = _mm_add_ps(t9, _mm_mul_ps(m1, n2));
  				t10 = _mm_add_ps(t10, _mm_mul_ps(m2, n2));
  				t11= _mm_add_ps(t11, _mm_mul_ps(m3, n2));
  				t12 = _mm_add_ps(t12, _mm_mul_ps(m0, n3));
  				t13 = _mm_add_ps(t13, _mm_mul_ps(m1, n3));
  				t14 = _mm_add_ps(t14, _mm_mul_ps(m2, n3));
  				t15 = _mm_add_ps(t15, _mm_mul_ps(m3, n3));
  				a0 += 1;
  				a1 += 1;
  				a2 += 1;
  				a3 += 1;
  			} while (a0 != end);
  			_mm_store_ps(&g.data[i][j], t0);
  			_mm_store_ps(&g.data[i+1][j], t1);
  			_mm_store_ps(&g.data[i+2][j], t2);
  			_mm_store_ps(&g.data[i+3][j], t3);
  			_mm_store_ps(&g.data[i][j+1], t4);
  			_mm_store_ps(&g.data[i+1][j+1], t5);
  			_mm_store_ps(&g.data[i+2][j+1], t6);
  			_mm_store_ps(&g.data[i+3][j+1], t7);
  			_mm_store_ps(&g.data[i][j+2], t8);
  			_mm_store_ps(&g.data[i+1][j+2], t9);
  			_mm_store_ps(&g.data[i+2][j+2], t10);
  			_mm_store_ps(&g.data[i+3][j+2], t11);
  			_mm_store_ps(&g.data[i][j+3], t12);
  			_mm_store_ps(&g.data[i+1][j+3], t13);
  			_mm_store_ps(&g.data[i+2][j+3], t14);
  			_mm_store_ps(&g.data[i+3][j+3], t15);
  }
  ```

  运行结果：

  ![Screenshot 2020-11-15 202832](E:\code\MatrixProduct\picture\Screenshot 2020-11-15 202832.png)

我不得不承认这是一次相当失败的尝试。。由于我对SSE指令集原理的无知直接导致了这一灾难性的结果。我相信老师稍微看一下我的代码便会发现里面存在很大问题，我也对于解决这些问题做出了相当长时间的努力，然而迫于多门期中考试的压力，最终还是没有能够拥有足够的时间去解决。

* 到目前为止对于1000 *1000的矩阵乘法我的优化最多也只能走到250ms左右，使用openblas大概需要33ms的时间，这其中差距太大。因此我不得不寻找新的优化方法。

* 经过不断尝试，我发现只需改变三重循环的循环次序便可使得访存连续，具体来说就是在做乘法时并不是按一行乘以列的顺序，而是先固定A的某一元素，将其与B的一整行元素相乘，所得结果加到目标矩阵的对应位置。这样做乘法并不能直接得到对应位置的值，只能得到该值的一部分，知道遍历完A中的元素最终才能得到所有结果。但这样做乘法能够使得访存连续。正因为这种使得访存连续的方式是如此的简单，所以它比我上面所提到的“packing”以及转置效果都要好。于是我在这种方法的基础上再进行了4 * 4的分块处理，便得到了简单而又高效的优化方法。具体代码如下：

  ```c++
  void product2(matrix& m, matrix& n, matrix& g)
  {
  	register float v0, v1, v2, v3;
  	register int mrow, mcol, ncol,i,j,k;
  	mrow = m.row;
  	mcol = m.col;
  	ncol = n.col;
  	for (i = 0; i < mrow; i++)
  	{
  		for (k = 0; k < mcol; k+=4)
  		{
  			v0 = m.data[i][k];
  			v1 = m.data[i][k + 1];
  			v2 = m.data[i][k + 2];
  			v3 = m.data[i][k + 3];
  			for (j = 0; j < ncol; j+=4) {
  				g.data[i][j] += v0 * n.data[k][j] + v1 * n.data[k + 1][j] + v2 * n.data[k + 2][j] + v3 * n.data[k + 3][j];
  				g.data[i][j+1] += v0 * n.data[k][j+1] + v1 * n.data[k + 1][j+1] + v2 * n.data[k + 2][j+1] + v3 * n.data[k + 3][j+1];
  				g.data[i][j+2] += v0 * n.data[k][j+2] + v1 * n.data[k + 1][j+2] + v2 * n.data[k + 2][j+2] + v3 * n.data[k + 3][j+2];
  				g.data[i][j+3] += v0 * n.data[k][j+3] + v1 * n.data[k + 1][j+3] + v2 * n.data[k + 2][j+3] + v3 * n.data[k + 3][j+3];
  			}
  		}
  	}
  }
  ```

  可以看到代码十分简洁甚至可以说简单。测试结果如下：

![Screenshot 2020-11-15 204405](E:\code\MatrixProduct\picture\Screenshot 2020-11-15 204405.png)

到此，我的优化便已超过了十倍。此后我又在该方法的基础上进行了诸多改进，均没有取得较为明显的效果。同时我还尝试使用openmp并行加速，但最终虽然速度变快了很多，结果却出现了极大的偏差，这种偏差已远远超出了误差的范围，故没有采用。

接下来是对所有方法的运行结果图，可以看到它们算出的结果都是一样的：

![Screenshot 2020-11-15 204854](E:\code\MatrixProduct\picture\Screenshot 2020-11-15 204854.png)

接下来是对10000 * 100000大小矩阵的测试。我将对比openblas和上述方法，分两次运行，结果不同，仅对比运行时间。

![image-20201115205413637](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20201115205413637.png)

![Screenshot 2020-11-15 210052](E:\code\MatrixProduct\picture\Screenshot 2020-11-15 210052.png)

可以看到当数据量变大时差距就显现出来了。openblas一般在30秒以内，而我的方法则需要三分钟才能计算出结果。当然，在数据量适中的时候，例如一百万的数据量，两者的差距较小，可忽略不计。

* 以上就是测试部分的全部内容。

## 二.程序实现：

* 本程序用于计算两个矩阵之间的乘法，主要特性如下：
   * 1.能够识别输入错误并给出对应的错误提示。
   * 2.最佳适用范围为大小1000 * 1000 以内的矩阵，数据量在该范围内的矩阵乘法有着较高的效率和准确性。与openblas的差距较小，且代码相当简洁易懂。
   * 3.对于数据量超过一亿的矩阵乘法需要的时间超过3分钟，时间较长，建议在这种数据量情况下直接使用openblas。
   * 4.注意：当矩阵数据量很小时，再使用分块不仅没有意义，反而显得十分繁琐。故对于数据量小于100 * 100的矩阵，程序将只使用改变循环次序的方法使得访存连续，而不使用分块的方法。在数据量较大时程序会自动使用分块的方法进行优化。
   
* 以下是代码实现：

   为了增加代码的可读性我将两个计算函数都放到了main.cpp里而不是头文件中。

   ```c++
   #include"matrix.h"
   using namespace std;
   void quickproduct(matrix& m, matrix& n, matrix& g);
   void blockproduct(matrix& m, matrix& n, matrix& g);
   int main()
   {
   	test();
   	bool b1(0), b2(0);
   	matrix m, n, g;
   	input(m,b1);
   	if (b1)
   	{
   		input(n, b2);
   	}
   	if(b2)
   	{
   		setmatrix(m.row, n.col, g);
   		for (int i = 0; i < m.row; i++)
   		{
   			for (int j = 0; j < n.col; j++)
   			{
   				g.data[i][j] = 0;
   			}
   		}
   		if (m.row * n.col < 10000)
   		{
   			quickproduct(m, n, g);
   		}
   		else
   		{
   			blockproduct(m, n, g);
   		}
   		for (int i = 0; i < m.row; i++)
   		{
   			for (int j = 0; j < n.col; j++)
   			{
   				cout << g.data[i][j] << " ";
   				if (j == n.col) cout << endl;
   			}
   		}
   	}
   }
   void quickproduct(matrix& m, matrix& n, matrix& g)
   {
   	for (int i = 0; i < m.row; i++)
   	{
   		for (int k = 0; k < m.col; k++)
   		{
   			int v = m.data[i][k];
   			for (int j = 0; j < n.col; j++) {
   				g.data[i][j] += v * n.data[k][j];
   			}
   		}
   	}
   }
   void blockproduct(matrix& m, matrix& n, matrix& g)
   {
   	register float v, v0, v1, v2, v3;
   	register int mrow, mcol, ncol, i, j, k;
   	mrow = m.row;
   	mcol = m.col;
   	ncol = n.col;
   	int x = mcol % 4;
   	int y = ncol % 4;
   	for (i = 0; i < mrow; i++)
   	{
   		for (k = 0; k < mcol - 3; k += 4)
   		{
   			v0 = m.data[i][k];
   			v1 = m.data[i][k + 1];
   			v2 = m.data[i][k + 2];
   			v3 = m.data[i][k + 3];
   			for (j = 0; j < ncol - 3; j += 4) {
   				g.data[i][j] += v0 * n.data[k][j] + v1 * n.data[k + 1][j] + v2 * n.data[k + 2][j] + v3 * n.data[k + 3][j];
   				g.data[i][j + 1] += v0 * n.data[k][j + 1] + v1 * n.data[k + 1][j + 1] + v2 * n.data[k + 2][j + 1] + v3 * n.data[k + 3][j + 1];
   				g.data[i][j + 2] += v0 * n.data[k][j + 2] + v1 * n.data[k + 1][j + 2] + v2 * n.data[k + 2][j + 2] + v3 * n.data[k + 3][j + 2];
   				g.data[i][j + 3] += v0 * n.data[k][j + 3] + v1 * n.data[k + 1][j + 3] + v2 * n.data[k + 2][j + 3] + v3 * n.data[k + 3][j + 3];
   			}
   			for (j = ncol - y; j < ncol; j++)
   			{
   				g.data[i][j] = v0 * n.data[k][j] + v1 * n.data[k + 1][j] + v2 * n.data[k + 2][j] + v3 * n.data[k + 3][j];
   			}
   		}
   		for (k = mcol - x; k < mcol; k++)
   		{
   			v = m.data[i][k];
   			for (j = 0; j < ncol; j++)
   			{
   				g.data[i][j] += v * n.data[k][j];
   			}
   		}
   	}
   }
   ```

   其中函数quickproduct是用来处理较小数据量的矩阵乘法，而blockproduct用来处理较大数据量的矩阵乘法。考虑到实际应用中并非所有的矩阵行，列长度均可被4整除，所以我在blockproduct中做出了相关处理。

   上述代码中的input函数用来输入矩阵，可以检测用户是否非法输入。该函数代码在头文件中，具体如下：

   ```c++
   bool input(matrix &u,bool &b)
   {
   	float x;
   	b=1;
   	cout << "Please input the lenth of row and column:";
   	cin >> u.row >> u.col;
   	cout << "Please input the elements of matrix:\n";
   	u.data = matrix_2d(u.row, u.col);
   	for (int i = 0; i < u.row; i++)
   	{
   		for (int j = 0; j < u.col; j++)
   		{
   			cin >> x;
   			if (!cin) {
   				cout << "please input real number!\n";
   				b = 0;
   			}
   			if (b == 0) break;
   			else u.data[i][j] = x;
   		}
   		if (b == 0) break;
   	}
   	if (b == 0) cout << "Input error!";
   	else cout << "input sucessfully!\n";
   	return b;
   }
   ```

* 接下来是对该程序的一些测试：

   ![image-20201115231612304](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20201115231612304.png)

   这是对于非法字符输入的处理。由于程序在接受矩阵长宽数值后会固定输入矩阵的数据量大小，所以不会存在输入数据过多或过少的情况。

* 接下来测试它结果的正确性。首先是较小数据量：

   

![Screenshot 2020-11-15 232105](E:\code\MatrixProduct\picture\Screenshot 2020-11-15 232105.png)

结果是正确的。接下来对于较大数据量，自然无法采取手动输入的方式，于是我将blockproduct移入头文件进行测试。测试矩阵A*B分别为数据量大小541 * 238 和238 * 541的矩阵，其中541和238均无法被4整除。用最原始的三重循环方法得到的结果与blockproduct进行对比。（取两者结果的第六行第六列的元素值进行对比）

![image-20201115232737996](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20201115232737996.png)

可以看到两者结果完全相同。注意由于我并没有进行O3优化，所以结果看上去有些慢。

* 至此应用程序的展示部分便已结束。可以看到对于适度数据量大小（一百万以内）范围内的矩阵乘法，本程序具有较好的效率与准确度。对于较大数据量（一亿以上）则运算较慢，需三分钟以上，应用价值不高。

  ## 三.总结

* 本次作业我成功将矩阵乘法的运算时间从6000ms缩减到78ms并保持答案一致。最终在两亿数据量的测试运行速度与openblas差距较大，但在一百万范围内相当接近。在作业过程中我尝试过各种各样的方法，扩展了知识面，收获颇丰。也遗留了一些问题，比如如何正确使用SSE指令集，以及如何正确使用openmp并行优化。这些问题我在本次作业中没有时间加以解决，但在之后的日子里我会思考，学习，并想出解决方案。提交作业后我依然会花时间更新github上的代码。希望到老师改到我作业的时候我已经解决了这些问题甚至想出了更好的办法^_^.

* 以上便是我期中project报告的全部内容。感谢阅读！
