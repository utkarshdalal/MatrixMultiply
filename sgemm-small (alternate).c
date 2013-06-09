#include <nmmintrin.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
void sgemm( int m, int n, float *A, float *C )
{
	__m128 a, a1, a2, a3, a4;
	__m128 b, b1, b2, b3, b4;
	__m128 c, c1, c2, c3, c4;
	int i, j, k, l;
	int mod = m%4;
	int end = m/4*4;
	int end16 = m/16 * 16;
	int total = n*m;
	int C_dim = m*m;
	int C_size = sizeof(C[0]);
	int C_bytes = C_dim * C_size;
	float num[4];
	float* A_address;
	memset(C, 0, C_bytes);
	#pragma omp parallel for private(a, a1, a2, a3, a4, b, b1, b2, b3, b4, c, c1, c2, c3, c4, i, j, k, l)
	for( i = 0; i < end16; i += 16 ){
		for( k = 0; k < total; k += m ) {
			a1 = _mm_loadu_ps(A + i + k);
			a2 = _mm_loadu_ps(A + i + k + 4);
			a3 = _mm_loadu_ps(A + i + k + 8);
			a4 = _mm_loadu_ps(A + i + k + 12);
			for( j = 0; j < end; j += 4 ) {

				b1 = _mm_load1_ps(A + k + j);
				c1 = _mm_loadu_ps(C + i + j*m);
				c2 = _mm_loadu_ps(C + i + j*m + 4);
				c3 = _mm_loadu_ps(C + i + j*m + 8);
				c4 = _mm_loadu_ps(C + i + j*m + 12);
				c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b1));
				c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b1));
				c3 = _mm_add_ps(c3, _mm_mul_ps(a3, b1));
				c4 = _mm_add_ps(c4, _mm_mul_ps(a4, b1));
				_mm_storeu_ps(C + i + j*m, c1);
				_mm_storeu_ps(C + i + j*m + 4, c2);
				_mm_storeu_ps(C + i + j*m + 8, c3);
				_mm_storeu_ps(C + i + j*m + 12, c4);

				b2 = _mm_load1_ps(A + k + j + 1);
				c1 = _mm_loadu_ps(C + i + j*m + m);
				c2 = _mm_loadu_ps(C + i + j*m + 4 + m);
				c3 = _mm_loadu_ps(C + i + j*m + 8 + m);
				c4 = _mm_loadu_ps(C + i + j*m + 12 + m);
				c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b2));
				c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b2));
				c3 = _mm_add_ps(c3, _mm_mul_ps(a3, b2));
				c4 = _mm_add_ps(c4, _mm_mul_ps(a4, b2));
				_mm_storeu_ps(C + i + j*m + m, c1);
				_mm_storeu_ps(C + i + j*m + 4 + m, c2);
				_mm_storeu_ps(C + i + j*m + 8 + m, c3);
				_mm_storeu_ps(C + i + j*m + 12 + m, c4);

				b3 = _mm_load1_ps(A + k + j + 2);
				c1 = _mm_loadu_ps(C + i + j*m + 2*m);
				c2 = _mm_loadu_ps(C + i + j*m + 4 + 2*m);
				c3 = _mm_loadu_ps(C + i + j*m + 8 + 2*m);
				c4 = _mm_loadu_ps(C + i + j*m + 12 + 2*m);
				c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b3));
				c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b3));
				c3 = _mm_add_ps(c3, _mm_mul_ps(a3, b3));
				c4 = _mm_add_ps(c4, _mm_mul_ps(a4, b3));
				_mm_storeu_ps(C + i + j*m + 2*m, c1);
				_mm_storeu_ps(C + i + j*m + 4 + 2*m, c2);
				_mm_storeu_ps(C + i + j*m + 8 + 2*m, c3);
				_mm_storeu_ps(C + i + j*m + 12 + 2*m, c4);

				b4 = _mm_load1_ps(A + k + j + 3);
				c1 = _mm_loadu_ps(C + i + j*m + 3*m);
				c2 = _mm_loadu_ps(C + i + j*m + 4 + 3*m);
				c3 = _mm_loadu_ps(C + i + j*m + 8 + 3*m);
				c4 = _mm_loadu_ps(C + i + j*m + 12 + 3*m);
				c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b4));
				c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b4));
				c3 = _mm_add_ps(c3, _mm_mul_ps(a3, b4));
				c4 = _mm_add_ps(c4, _mm_mul_ps(a4, b4));
				_mm_storeu_ps(C + i + j*m + 3*m, c1);
				_mm_storeu_ps(C + i + j*m + 4 + 3*m, c2);
				_mm_storeu_ps(C + i + j*m + 8 + 3*m, c3);
				_mm_storeu_ps(C + i + j*m + 12 + 3*m, c4);
			}
		}
	}//Looks about right to me for a matrix where m is divisible by 4.

	for( i = end16; i < end; i += 4 ){
		for( k = 0; k < total; k += m ) {
			a = _mm_loadu_ps(A + i + k);
			for( j = 0; j < m; j += 1 ) {
				b = _mm_load1_ps(A + k + j);
				c = _mm_loadu_ps(C + i + j*m);
				c = _mm_add_ps(c, _mm_mul_ps(a, b));
				_mm_storeu_ps(C + i + j*m, c);
			}
		}
	}
	if (mod != 0){
		if (mod == 3){
			for( i = end; i < m; i +=4 ){
				for( k = 0; k < m; k++ ) {
					A_address = A + i;
					c = _mm_setzero_ps();
					for( j = 0; j < total; j += m ) {
						a = _mm_setr_ps(*(A_address),*(A_address + 1),*(A_address + 2), 0);
						b = _mm_load1_ps(A + k + j);
						c = _mm_add_ps(c, _mm_mul_ps(a, b));
						A_address += m;
					}
					_mm_storeu_ps(num, c);
					for (l = 0; l < 3; l ++){
						*(C + i + k*m + l) = num[l];
					}
				}
			}
		}
		else if (mod == 2){
			for( i = end; i < m; i +=4 ){
				for( k = 0; k < m; k++ ) {
					A_address = A + i;
					c = _mm_setzero_ps();
					for( j = 0; j < total; j += m ) {
						a = _mm_setr_ps(*(A_address),*(A_address + 1), 0 ,0);
						b = _mm_load1_ps(A + k + j);
						c = _mm_add_ps(c, _mm_mul_ps(a, b));
						A_address += m;
					}
					_mm_storeu_ps(num, c);
					for (l = 0; l < 2; l ++){
						*(C + i + k*m + l) = num[l];
					}
				}
			}
		}
		else if (mod == 1){
			for( i = end; i < m; i +=4 ){
				for( k = 0; k < m; k++ ) {
					A_address = A + i;
					c = _mm_setzero_ps();
					for( j = 0; j < total; j += m ) {
						a = _mm_setr_ps(*(A_address), 0, 0, 0);
						b = _mm_load1_ps(A + k + j);
						c = _mm_add_ps(c, _mm_mul_ps(a, b));
						A_address += m;
					}
					_mm_storeu_ps(num, c);
					for (l = 0; l < 1; l ++){
						*(C + i + k*m + l) = num[l];
					}
				}
			}
		}
	}
}
