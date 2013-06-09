#include <nmmintrin.h>
#include <omp.h>
void sgemm( int m, int n, float *A, float *C )
{
    __m128 a1, a2, a3, a4, a5, a6, a7, a8;    
    __m128 b1, b2, b3, b4;    
    __m128 c;
    
    int i, j, k, x, y, z;
    int mod = m%4;
    int end = m/4 * 4;
	int endm64 = m/64 * 64;
	int endn64 = n/64 * 64;
    int total = n*m;
    float num[4];
    float* A_address;
	float* B_address;
    float* C_address;
    int m3 = 3 * m;
    int m2 = 2 * m;
    int end1 = total/m3 * m3;
#pragma omp parallel for private(a, a1, a2, a3, b, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, c, c1, c2, c3, c4, i, j, k, l)
	for(x = 0; x < endm64; x += 64){
		for(y = 0; y < endn64; y += 64){
			for(z = 0; z < endm64; z += 64){
    			for(i = A + z + y*m; i < N; i += 8){
					for(j = A + x + y*m; j < end; j += 8) {
	    				for(k = C + z + x*m; k < end1; k += 8){
							A_address = k + i * m;

							a1 = _mm_loadu_ps(A_address);				
							a2 = _mm_loadu_ps(A_address + m);				
							a3 = _mm_loadu_ps(A_address + m2);				
							a4 = _mm_loadu_ps(A_address + m3);
							a5 = _mm_loadu_ps(A_address + 4);
							a6 = _mm_loadu_ps(A_address + m + 4);
							a7 = _mm_loadu_ps(A_address + m2 + 4);
							a8 = _mm_loadu_ps(A_address + m3 + 4);

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							/*

							REPEAT

							*/

							A_address += 4*m;
							B_address -= 7;
							B_address += 4*m;
							C_address -= 7*m;

							a1 = _mm_loadu_ps(A_address);				
							a2 = _mm_loadu_ps(A_address + m);				
							a3 = _mm_loadu_ps(A_address + m2);				
							a4 = _mm_loadu_ps(A_address + m3);
							a5 = _mm_loadu_ps(A_address + 4);
							a6 = _mm_loadu_ps(A_address + m + 4);
							a7 = _mm_loadu_ps(A_address + m2 + 4);
							a8 = _mm_loadu_ps(A_address + m3 + 4);

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

							B_address += 1;
							C_address += m;

							b1 = _mm_load1_ps(B_address);
							b2 = _mm_load1_ps(B_address + m);
							b3 = _mm_load1_ps(B_address + m2);
							b4 = _mm_load1_ps(B_address + m3);

							c = _mm_loadu_ps(C_address);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a1));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a2));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a3));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a4));
							_mm_storeu_ps(C_address, c);

							c = _mm_loadu_ps(C_address + 4);
							c = _mm_add_ps(c, _mm_mul_ps(b1, a5));
							c = _mm_add_ps(c, _mm_mul_ps(b2, a6));
							c = _mm_add_ps(c, _mm_mul_ps(b3, a7));
							c = _mm_add_ps(c, _mm_mul_ps(b4, a8));
							_mm_storeu_ps(C_address + 4, c);

	}
	for(k = end; k < m; k++){
	    float* A_address1 = A + i;
	    float* A_address2 = A + k;
	    c = _mm_setzero_ps();
	    for( j = 0; j < end1; j += m3, A_address1 += m3, A_address2 += m3){
		a1 = _mm_loadu_ps(A_address1);
		a2 = _mm_loadu_ps(A + i + j + m);
		a3 = _mm_loadu_ps(A + i + j + m2);
		
		b1 = _mm_load1_ps(A_address2);
		b2 = _mm_load1_ps(A + k + j + m);
		b3 = _mm_load1_ps(A + k + j + m2);
		
		c = _mm_add_ps(c, _mm_mul_ps(a1, b1));
		c = _mm_add_ps(c, _mm_mul_ps(a2, b2));
		c = _mm_add_ps(c, _mm_mul_ps(a3, b3));
	    }
	    for( j = end1; j < total; j += m){
		a = _mm_loadu_ps(A + i + j);
		
		b = _mm_load1_ps(A + k + j);
		
		c = _mm_add_ps(c, _mm_mul_ps(a, b));
	    }
	    _mm_storeu_ps(C + i + k*m, c);
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
