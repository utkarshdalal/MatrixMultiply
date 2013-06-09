#include <nmmintrin.h>
void sgemm( int m, int n, float *A, float *C )
{
    __m128 a;
    __m128 a1;
    __m128 a2;
    __m128 a3;
    __m128 a4;
    /*
    __m128 a5;
    __m128 a6;
    __m128 a7;
    __m128 a8;
    */
    
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
    __m128 b13;
    __m128 b14;
    __m128 b15;
    __m128 b16;
    
    __m128 c;
    __m128 c1;
    __m128 c2;
    __m128 c3;
    __m128 c4;
    
    int i, j, k, l;
    int mod = m%4;
    int end = m/4*4;
    int total = n*m;
    int m4 = m*4;
    int m4mod = total / m4 * m4;
    float num[4];
    float* A_address, A_address1, A_address2;
    for( i = 0; i < end; i +=4 ){
	for( k = 0; k < m / 4 * 4; k+= 4 ) {
	    c1 = _mm_setzero_ps();
	    c2 = _mm_setzero_ps();
	    c3 = _mm_setzero_ps();
	    c4 = _mm_setzero_ps();
	    A_address1 = A + i;
	    A_address2 = A + k;
	    for( j = 0; j < m4mod; j += m4, A_address1 += m4, A_address2 += m4){
		a1 = _mm_loadu_ps(A_address1);
		a2 = _mm_loadu_ps(A_address1 + m);
		a3 = _mm_loadu_ps(A_address1 + 2*m);
		a4 = _mm_loadu_ps(A_address1 + 3*m);
		
		/*
		a5 = _mm_loadu_ps(A_address1 + 4*m);
	        a6 = _mm_loadu_ps(A_address1 + 5*m);
		a7 = _mm_loadu_ps(A_address1 + 6*m);
		a8 = _mm_loadu_ps(A_address1 + 7*m);
		*/
		
		b1 = _mm_load1_ps(A_address2);
		b2 = _mm_load1_ps(A_address2 + m);
		b3 = _mm_load1_ps(A_address2 + 2*m);
		b4 = _mm_load1_ps(A_address2 + 3*m);
		
		b5 = _mm_load1_ps(A_address2 + 1);
		b6 = _mm_load1_ps(A_address2 + m + 1);
		b7 = _mm_load1_ps(A_address2 + 2*m + 1);
		b8 = _mm_load1_ps(A_address2 + 3*m + 1);
		
		b9 = _mm_load1_ps(A_address2 + 2);
		b10 = _mm_load1_ps(A_address2 + m + 2);
		b11 = _mm_load1_ps(A_address2 + 2*m + 2);
		b12 = _mm_load1_ps(A_address2 + 3*m + 2);
		
		b13 = _mm_load1_ps(A_address2 + 3);
		b14 = _mm_load1_ps(A_address2 + m + 3);
		b15 = _mm_load1_ps(A_address2 + 2*m + 3);
		b16 = _mm_load1_ps(A_address2 + 3*m + 3);
		
		c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b1));
		c1 = _mm_add_ps(c1, _mm_mul_ps(a2, b2));
		c1 = _mm_add_ps(c1, _mm_mul_ps(a3, b3));
		c1 = _mm_add_ps(c1, _mm_mul_ps(a4, b4));
		
		c2 = _mm_add_ps(c2, _mm_mul_ps(a1, b5));
		c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b6));
		c2 = _mm_add_ps(c2, _mm_mul_ps(a3, b7));
		c2 = _mm_add_ps(c2, _mm_mul_ps(a4, b8));
		
		c3 = _mm_add_ps(c3, _mm_mul_ps(a1, b9));
		c3 = _mm_add_ps(c3, _mm_mul_ps(a2, b10));
		c3 = _mm_add_ps(c3, _mm_mul_ps(a3, b11));
		c3 = _mm_add_ps(c3, _mm_mul_ps(a4, b12));
		
		c4 = _mm_add_ps(c4, _mm_mul_ps(a1, b13));
		c4 = _mm_add_ps(c4, _mm_mul_ps(a2, b14));
		c4 = _mm_add_ps(c4, _mm_mul_ps(a3, b15));
		c4 = _mm_add_ps(c4, _mm_mul_ps(a4, b16));
		
	    }
	    A_address1 = A + i;
	    for( j = m4mod; j < total; j += m, A_address1 += m){
			a = _mm_loadu_ps(A_address1);
		
			b1 = _mm_load1_ps(A_address1);
			b2 = _mm_load1_ps(A_address1 + 1);
			b3 = _mm_load1_ps(A_address1 + 2);
			b4 = _mm_load1_ps(A_address1 + 3);
		
			c1 = _mm_add_ps(c1, _mm_mul_ps(a, b1));
			c2 = _mm_add_ps(c2, _mm_mul_ps(a, b2));
			c3 = _mm_add_ps(c3, _mm_mul_ps(a, b3));
			c4 = _mm_add_ps(c4, _mm_mul_ps(a, b4));
	    }
	    _mm_storeu_ps(C + i + k*m, c1);
	    _mm_storeu_ps(C + i + (k+1)*m, c2);
	    _mm_storeu_ps(C + i + (k+2)*m, c3);
	    _mm_storeu_ps(C + i + (k+3)*m, c4);
	}
	A_address1 = A + i;
	A_address2 = A + k;
	for(k = end; k < m; k++){
	    c = _mm_setzero_ps();
	    for( j = 0; j < m4mod; j += m4, A_address1 += m4, A_address2 += m4){
		a1 = _mm_loadu_ps(A_address1);
		a2 = _mm_loadu_ps(A_address1 + m);
		a3 = _mm_loadu_ps(A_address1 + 2*m);
		a4 = _mm_loadu_ps(A_address1 + 3*m);

		/*
		a5 = _mm_loadu_ps(A_address1 + 4*m);
		a6 = _mm_loadu_ps(A_address1 + 5*m);
		a7 = _mm_loadu_ps(A_address1 + 6*m);
		a8 = _mm_loadu_ps(A_address1 + 7*m);
		*/
		
		b1 = _mm_load1_ps(A_address2);
		b2 = _mm_load1_ps(A_address2 + m);
		b3 = _mm_load1_ps(A_address2 + 2*m);
		b4 = _mm_load1_ps(A_address2 + 3*m);

		/*
		b5 = _mm_load1_ps(A_address2 + 1);
		b6 = _mm_load1_ps(A_address2 + m + 1);
		b7 = _mm_load1_ps(A_address2 + 2*m + 1);
		b8 = _mm_load1_ps(A_address2 + 3*m + 1);
		
		b9 = _mm_load1_ps(A_address2 + 2);
		b10 = _mm_load1_ps(A_address2 + m + 2);
		b11 = _mm_load1_ps(A_address2 + 2*m + 2);
		b12 = _mm_load1_ps(A_address2 + 3*m + 2);
		*/
		
		c = _mm_add_ps(c, _mm_mul_ps(a1, b1));
		c = _mm_add_ps(c, _mm_mul_ps(a2, b2));
		c = _mm_add_ps(c, _mm_mul_ps(a3, b3));
		c = _mm_add_ps(c, _mm_mul_ps(a4, b4));
	    }
	    for( j = m4mod; j < total; j += m){
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
