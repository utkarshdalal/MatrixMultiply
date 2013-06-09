#include <nmmintrin.h>
void sgemm( int m, int n, float *A, float *C )
{
	__m128 a;
	__m128 b;
	__m128 c;
	int i, j, k, l;
	int mod = m%4;
	int end = m/4*4;
	int total = n*m;
	float num[4];
	float* A_address;
	for( i = 0; i < end; i +=4 ){
		for( k = 0; k < m; k++ ) {
			c = _mm_setzero_ps();
			for( j = 0; j < total; j += m ) {
				a = _mm_loadu_ps(A + i + j);
				b = _mm_load1_ps(A + k + j);
				c = _mm_add_ps(c, _mm_mul_ps(a, b));
			}
			_mm_storeu_ps(C + i + k*m, c);
		}
	}//Looks about right to me for a matrix where m is divisible by 4.
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
						a = _mm_setr_ps(*(A_address),*(A_address + 1),0 ,0);
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
