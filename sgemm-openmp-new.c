#include <nmmintrin.h>
#include <string.h>
void sgemm( int m, int n, float *A, float *C )
{
    __m128 a, a2, a3, a4;
	__m128 b, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16;
	__m128 c, c1, c2, c3, c4;
	int i, j, k, l;
	int mod = m%4;
	int end = m/4*4;
	int total = n*m; 
	int C_dim = m*m;
	int C_size = sizeof(C[0]);
	int C_bytes = C_dim * C_size;
	float num[4];
	float* A_address;
	//C[C_dim] = {0.0};
	//bzero(C, C_bytes);
	memset(C, 0, C_bytes);
#pragma omp parallel for private(i, j, k, a, b, c, a1, a2, a3, a4, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, c1, c2, c3, c4)
	for( i = 0; i < end; i += 4 ){
		for(k = 0; k < total / 4 * 4; k += 4*m) {
			a1 = _mm_loadu_ps(A + i + k);
			a2 = _mm_loadu_ps(A + i + k + m);
			a3 = _mm_loadu_ps(A + i + k + 2*m);
			a4 = _mm_loadu_ps(A + i + k + 3*m);

			for(j = 0; j < m / 4 * 4; j += 4){
			    b1 = _mm_load1_ps(A + k + j);
			    b2 = _mm_load1_ps(A + k + j + 1);
			    b3 = _mm_load1_ps(A + k + j + 2);
			    b4 = _mm_load1_ps(A + k + j + 3);
			    
			    b5 = _mm_load1_ps(A + k + m + j);
			    b6 = _mm_load1_ps(A + k + m + j + 1);
			    b7 = _mm_load1_ps(A + k + m + j + 2);
			    b8 = _mm_load1_ps(A + k + m + j + 3);
			    
			    b9 = _mm_load1_ps(A + k + 2*m + j);
			    b10 = _mm_load1_ps(A + k + 2*m + j + 1);
			    b11 = _mm_load1_ps(A + k + 2*m + j + 2);
			    b12 = _mm_load1_ps(A + k + 2*m + j + 3);
			    
			    b13 = _mm_load1_ps(A + k + 3*m + j);
			    b14 = _mm_load1_ps(A + k + 3*m + j + 1);
			    b15 = _mm_load1_ps(A + k + 3*m + j + 2);
			    b16 = _mm_load1_ps(A + k + 3*m + j + 3);
			    
			    c1 = _mm_loadu_ps(C + i + j*m);
			    c2 = _mm_loadu_ps(C + i + (j+1)*m);
			    c3 = _mm_loadu_ps(C + i + (j+2)*m);
			    c4 = _mm_loadu_ps(C + i + (j+3)*m);
			    
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b1));
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a2, b5));
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a3, b9));
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a4, b13));

			    /*
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a2, b2));
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a2, b6));
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a2, b10));
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a2, b14));
			        
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a3, b3));
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a3, b7));
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a3, b11));
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a3, b15));

			    c1 = _mm_add_ps(c1, _mm_mul_ps(a4, b4));
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a4, b8));
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a4, b12));
			    c1 = _mm_add_ps(c1, _mm_mul_ps(a4, b16));
			    */
			    
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a1, b2));
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b6));
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a3, b10));
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a4, b14));

			    /*
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b2));
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b6));
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b10));
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b14));
			    
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a3, b3));
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a3, b7));
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a3, b11));
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a3, b15));
			    
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a4, b4));
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a4, b8));
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a4, b12));
			    c2 = _mm_add_ps(c2, _mm_mul_ps(a4, b16));
			    */
			    
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a1, b3));
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a2, b7));
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a3, b11));
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a4, b15));

			    /*
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a2, b2));
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a2, b6));
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a2, b10));
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a2, b14));

			    c3 = _mm_add_ps(c3, _mm_mul_ps(a3, b3));
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a3, b7));
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a3, b11));
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a3, b15));

			    c3 = _mm_add_ps(c3, _mm_mul_ps(a4, b4));
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a4, b8));
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a4, b12));
			    c3 = _mm_add_ps(c3, _mm_mul_ps(a4, b16));
			    */
			    
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a1, b4));
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a2, b8));
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a3, b12));
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a4, b16));

			    /*
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a2, b2));
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a2, b6));
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a2, b10));
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a2, b14));
			        
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a3, b3));
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a3, b7));
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a3, b11));
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a3, b15));
			        
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a4, b4));
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a4, b8));
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a4, b12));
			    c4 = _mm_add_ps(c4, _mm_mul_ps(a4, b16));
			    */
			    
			    _mm_storeu_ps(C + i + j*m, c1);
			    _mm_storeu_ps(C + i + (j+1)*m, c2);
			    _mm_storeu_ps(C + i + (j+2)*m, c3);
			    _mm_storeu_ps(C + i + (j+3)*m, c4);
			}
			for(j = m / 4 * 4; j < m; j++){
			    b1 = _mm_load1_ps(A + k + j);
			    b2 = _mm_load1_ps(A + k + m + j);
			    b3 = _mm_load1_ps(A + k + 2*m + j);
			    b4 = _mm_load1_ps(A + k + 3*m + j);
			    c = _mm_loadu_ps(C + i + j*m);
			    
			    c = _mm_add_ps(c, _mm_mul_ps(a1, b1));
			    c = _mm_add_ps(c, _mm_mul_ps(a2, b2));
			    c = _mm_add_ps(c, _mm_mul_ps(a3, b3));
			    c = _mm_add_ps(c, _mm_mul_ps(a4, b4));

			    /*
			    c = _mm_add_ps(c, _mm_mul_ps(a2, b1));
			    c = _mm_add_ps(c, _mm_mul_ps(a2, b2));
			    c = _mm_add_ps(c, _mm_mul_ps(a2, b3));
			    c = _mm_add_ps(c, _mm_mul_ps(a2, b4));

			    c = _mm_add_ps(c, _mm_mul_ps(a3, b1));
			    c = _mm_add_ps(c, _mm_mul_ps(a3, b2));
			    c = _mm_add_ps(c, _mm_mul_ps(a3, b3));
			    c = _mm_add_ps(c, _mm_mul_ps(a3, b4));

			    c = _mm_add_ps(c, _mm_mul_ps(a4, b1));
			    c = _mm_add_ps(c, _mm_mul_ps(a4, b2));
			    c = _mm_add_ps(c, _mm_mul_ps(a4, b3));
			    c = _mm_add_ps(c, _mm_mul_ps(a4, b4));
			    */
			    
			    _mm_storeu_ps(C + i + j*m, c);
			}
		}
		for(k = total / 4 * 4; k < total; k += m){
		    a = _mm_loadu_ps(A + i + k);
		    for(j = 0; j < m / 4 * 4; j += 4){
			b1 = _mm_load1_ps(A + k + j);
			b2 = _mm_load1_ps(A + k + j + 1);
			b3 = _mm_load1_ps(A + k + j + 2);
			b4 = _mm_load1_ps(A + k + j + 3);
			
			c1 = _mm_loadu_ps(C + i + j*m);
			c2 = _mm_loadu_ps(C + i + (j+1)*m);
			c3 = _mm_loadu_ps(C + i + (j+2)*m);
			c4 = _mm_loadu_ps(C + i + (j+3)*m);
			
			c1 = _mm_add_ps(c1, _mm_mul_ps(a, b1));
			c2 = _mm_add_ps(c2, _mm_mul_ps(a, b2));
			c3 = _mm_add_ps(c3, _mm_mul_ps(a, b3));
			c4 = _mm_add_ps(c4, _mm_mul_ps(a, b4));
			
			_mm_storeu_ps(C + i + j*m, c1);
			_mm_storeu_ps(C + i + (j+1)*m, c2);
			_mm_storeu_ps(C + i + (j+2)*m, c3);
			_mm_storeu_ps(C + i + (j+3)*m, c4);
		    }
		    for(j = m / 4 * 4; j < m; j++){
			b = _mm_load1_ps(A + k + j);
			c = _mm_loadu_ps(C + i + j*m);
			c = _mm_add_ps(c, _mm_mul_ps(a, b));
			_mm_storeu_ps(C + i + j*m, c);
		    }
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
