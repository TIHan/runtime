// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

.assembly extern legacy library mscorlib {}


.class public		conv_ovf_u4 {



.method public static		int32 conv(int64,int32) {
.locals		(class [mscorlib]System.OverflowException,int32)
.maxstack	2
try_start:
	ldarg		0
	conv.ovf.u4
	ldarg		1
	ceq
	stloc.1
	leave.s	try_end
try_end:
	ldloc.1
	brfalse			FAIL
	ldc.i4			0x11111111
	br				END
aHandler:
	isinst			[mscorlib]System.OverflowException
	stloc	0
	leave			HEnd
HEnd:

	ldloc	0
	brfalse			FAIL
	ldc.i4			0xEEEEEEEE
	br				END
FAIL:
	ldc.i4			0x00000000
	br				END
END:
	ret
.try try_start to try_end catch [mscorlib]System.OverflowException handler aHandler to HEnd
}

.method public static		int32 conv_un(int64,int32) {
.locals		(class [mscorlib]System.OverflowException,unsigned int32)
.maxstack	2
try_start:
	ldarg		0
	conv.ovf.u4.un
	stloc.1
	leave.s	try_end
try_end:
	ldloc.1
	ldarg		1
	ceq
	brfalse			FAIL
	ldc.i4			0x11111111
	br				END
aHandler:
	isinst			[mscorlib]System.OverflowException
	stloc	0
	leave			HEnd
HEnd:

	ldloc	0
	brfalse			FAIL
	ldc.i4			0xEEEEEEEE
	br				END
FAIL:
	ldc.i4			0x00000000
	br				END
END:
	ret
.try try_start to try_end catch [mscorlib]System.OverflowException handler aHandler to HEnd
}


.method public		void conv_ovf_u4() {
.maxstack	1
	ret
}

.method public static	int32 main(class [mscorlib]System.String[]) {
.entrypoint
.maxstack		5
	ldc.i8			0x0000000000000000
	ldc.i4			0x00000000
	call		int32 conv_ovf_u4::conv(int64,int32)
	ldc.i4			0x11111111
	ceq
	brfalse			FAIL
	ldc.i8			0x0000000000000001
	ldc.i4			0x00000001
	call		int32 conv_ovf_u4::conv(int64,int32)
	ldc.i4			0x11111111
	ceq
	brfalse			FAIL
	ldc.i8			0xFFFFFFFFFFFFFFFE
	ldc.i4			0x00000000
	call		int32 conv_ovf_u4::conv(int64,int32)
	ldc.i4			0xEEEEEEEE
	ceq
	brfalse			FAIL
	ldc.i8			0xFFFFFFFFFFFFFFFF
	ldc.i4			0x00000000
	call		int32 conv_ovf_u4::conv(int64,int32)
	ldc.i4			0xEEEEEEEE
	ceq
	brfalse			FAIL

	ldc.i8			0xFFFFFFFF
	ldc.i4			0xFFFFFFFF
	call		int32 conv_ovf_u4::conv(int64,int32)
	ldc.i4			0x11111111
	ceq
	brfalse			FAIL

	ldc.i8			0x0000000000000000
	ldc.i4			0x00000000
	call		int32 conv_ovf_u4::conv_un(int64,int32)
	ldc.i4			0x11111111
	ceq
	brfalse			FAIL
	ldc.i8			0x0000000000000001
	ldc.i4			0x00000001
	call		int32 conv_ovf_u4::conv_un(int64,int32)
	ldc.i4			0x11111111
	ceq
	brfalse			FAIL
	ldc.i8			0xFFFFFFFFFFFFFFFE
	ldc.i4			0x00000000
	call		int32 conv_ovf_u4::conv_un(int64,int32)
	ldc.i4			0xEEEEEEEE
	ceq
	brfalse			FAIL
	ldc.i8			0xFFFFFFFFFFFFFFFF
	ldc.i4			0x00000000
	call		int32 conv_ovf_u4::conv_un(int64,int32)
	ldc.i4			0xEEEEEEEE
	ceq
	brfalse			FAIL

	ldc.i8			0xFFFFFFFF
	ldc.i4			0xFFFFFFFF
	call		int32 conv_ovf_u4::conv_un(int64,int32)
	ldc.i4			0x11111111
	ceq
	brfalse			FAIL

PASS:
	ldc.i4 100
	br		END

FAIL:
	ldc.i4	0x00000000

END:
	ret
}
}

.assembly conv_ovf_u8_u4{}
