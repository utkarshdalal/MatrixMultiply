#include <nmmintrin.h>
#include <omp.h>
void sgemm( int m, int n, float *A, float *C )
{
    __m128 a;
    __m128 a1;
    __m128 a2; 
    __m128 a3;
    __m128 a4;
    __m128 a5;
    
    __m128 b;
    __m128 b1;
    __m128 b2;
    __m128 b3;
    __m128 b4;
    __m128 b5;
    __m128 b6;
    __m128 b7;
    __m128 b8;
    __m128 b9;
    __m128 b10;
    __m128 b11;
    __m128 b12;
    /*
    __m128 b13;
    __m128 b14;
    __m128 b15;
    __m128 b16;
    __m128 b17;
    __m128 b18;
    __m128 b19;
    __m128 b20;
    */
    
    __m128 c;
    __m128 c1;
    __m128 c2;
    __m128 c3;
    __m128 c4;
    
    int i, j, k, l;
    int mod = m%4;
    int end = m/4 * 4;
    int total = n*m;
    float num[4];
    float* A_address;
    float* C_address;
    int m3 = 3 * m;
    int m2 = 2 * m;
    int end1 = total/m3 * m3;
#pragma omp parallel for private(a, a1, a2, a3, b, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, c, c1, c2, c3, c4, i, j, k, l)
    for( i = 0; i < end; i += 4 ){
	for( k = 0; k < end; k += 4 ) {
	    c1 = _mm_setzero_ps();
	    c2 = _mm_setzero_ps();
	    c3 = _mm_setzero_ps();
	    c4 = _mm_setzero_ps();
	    float* A_address1 = A + i;
	    float* A_address2 = A + k;
	    float* A_address21 = A + k + 1;
	    for( j = 0; j < end1; j += m3, A_address1 += m3, A_address2 += m3, A_address21 += m3){
		a1 = _mm_loadu_ps(A_address1);
		a2 = _mm_loadu_ps(A_address1 + m);
		a3 = _mm_loadu_ps(A_address1 + m2);
		
		b1 = _mm_load1_ps(A_address2);
		b2 = _mm_load1_ps(A_address2 + m);
		b3 = _mm_load1_ps(A_address2 + m2);
		/*
		b4 = _mm_load1_ps(A_address2 + m3);
		b5 = _mm_load1_ps(A_address2 + m4);
		*/
		
		b4 = _mm_load1_ps(A_address21);
		b5 = _mm_load1_ps(A_address21 + m);
		b6 = _mm_load1_ps(A_address21 + m2);
		/*
		b9 = _mm_load1_ps(A_address21 + m3);
		b10 = _mm_load1_ps(A_address21 + m4);
		*/
		b7 = _mm_load1_ps(A + k + 2 + j);
		b8 = _mm_load1_ps(A + k + 2 + j + m);
		b9 = _mm_load1_ps(A + k + 2 + j + m2);
		/*
		b14 = _mm_load1_ps(A + k + 2 + j + m3);
		b15 = _mm_load1_ps(A + k + 2 + j + m4);
		*/
		
		b10 = _mm_load1_ps(A + k + 3 + j);
		b11 = _mm_load1_ps(A + k + 3 + j + m);
		b12 = _mm_load1_ps(A + k + 3 + j + m2);
		/*
		b19 = _mm_load1_ps(A + k + 3 + j + m3);
		b20 = _mm_load1_ps(A + k + 3 + j + m4);
		*/
		
		c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b1));
		c1 = _mm_add_ps(c1, _mm_mul_ps(a2, b2));
		c1 = _mm_add_ps(c1, _mm_mul_ps(a3, b3));
		/*
		c1 = _mm_add_ps(c1, _mm_mul_ps(a4, b4));
		c1 = _mm_add_ps(c1, _mm_mul_ps(a5, b5));
		*/
		c2 = _mm_add_ps(c2, _mm_mul_ps(a1, b4));
		c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b5));
		c2 = _mm_add_ps(c2, _mm_mul_ps(a3, b6));
		/*
		c2 = _mm_add_ps(c2, _mm_mul_ps(a4, b9));
		c2 = _mm_add_ps(c2, _mm_mul_ps(a5, b10));
		*/
		c3 = _mm_add_ps(c3, _mm_mul_ps(a1, b7));
		c3 = _mm_add_ps(c3, _mm_mul_ps(a2, b8));
		c3 = _mm_add_ps(c3, _mm_mul_ps(a3, b9));
		/*
		c3 = _mm_add_ps(c3, _mm_mul_ps(a4, b14));
		c3 = _mm_add_ps(c3, _mm_mul_ps(a5, b15));
		*/
		
		c4 = _mm_add_ps(c4, _mm_mul_ps(a1, b10));
		c4 = _mm_add_ps(c4, _mm_mul_ps(a2, b11));
		c4 = _mm_add_ps(c4, _mm_mul_ps(a3, b12));
		/*
		c4 = _mm_add_ps(c4, _mm_mul_ps(a4, b19));
		c4 = _mm_add_ps(c4, _mm_mul_ps(a5, b20));
		*/
		
	    }
	    for( j = end1; j < total; j += m){
		a = _mm_loadu_ps(A + i + j);
		
		b1 = _mm_load1_ps(A + k + j);
		b2 = _mm_load1_ps(A + k + 1 + j);
		b3 = _mm_load1_ps(A + k + 2 + j);
		b4 = _mm_load1_ps(A + k + 3 + j);
		
		c1 = _mm_add_ps(c1, _mm_mul_ps(a, b1));
		c2 = _mm_add_ps(c2, _mm_mul_ps(a, b2));
		c3 = _mm_add_ps(c3, _mm_mul_ps(a, b3));
		c4 = _mm_add_ps(c4, _mm_mul_ps(a, b4));
	    }
	    _mm_storeu_ps(C + i + (k)*m, c1);
	    _mm_storeu_ps(C + i + (k+1)*m, c2);
	    _mm_storeu_ps(C + i + (k+2)*m, c3);
	    _mm_storeu_ps(C + i + (k+3)*m, c4);
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
