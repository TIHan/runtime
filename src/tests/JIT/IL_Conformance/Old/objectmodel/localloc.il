// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.


.assembly extern System.Console
{
  .publickeytoken = (B0 3F 5F 7F 11 D5 0A 3A )
  .ver 4:0:0:0
}

.assembly extern legacy library mscorlib {}
.assembly 'localloc'{
}

.class public			_localloc {

.method public void .ctor() {

.maxstack	10
	ldarg.0
	call instance void [mscorlib]System.Object::.ctor() 
	ret
}

.method public static	int32 main(string[]) {
.locals		(class [mscorlib]System.IO.TextWriter,int32*,int32,class [mscorlib]System.Text.StringBuilder,int32,string)
.entrypoint
	call		class [mscorlib]System.IO.TextWriter [System.Console]System.Console::get_Out()
	stloc		0
	ldc.i4		0x0
	stloc		2
	newobj	instance	void [mscorlib]System.Text.StringBuilder::.ctor() 
	stloc		3

TryStart:
	ldc.i4		0xFF
	localloc
	stloc		1

LOOP0:
	ldloc		2
	ldc.i4		0xFF
	bgt			END0

	ldloc		1
	ldloc		2
	add

	ldloc		2
	conv.i1
	stind.i1

	ldc.i4		0x1
	ldloc		2
	add
	stloc		2

	br			LOOP0

END0:

	ldc.i4		0x0
	stloc		2

LOOP1:
	ldloc		2
	ldc.i4		0xFF
	bgt			PassInLoop
	
	ldloc		1
	ldloc		2
	add

	ldind.i1

	ldloc		2
	conv.i1

	bne.un		FailInLoop

	ldc.i4		0x1
	ldloc		2
	add
	stloc		2

	br			LOOP1

END1:

	br		BeforeTryEnd

FailInLoop:
	leave		Fail
PassInLoop:
	leave		Pass

BeforeTryEnd:
leave Pass
TryEnd:
	br			Pass

Filter:
	pop
	ldc.i4		0x1
	endfilter

Handler:
	pop
	leave			EndHandler
EndHandler:
	br			Fail

Pass:
	ldloc			0
	ldstr			"LOCALLOC Tests PASS"


	callvirt	instance	void [mscorlib]System.IO.TextWriter::WriteLine(string)
	ldc.i4 100
	br			End

Fail:
	ldloc		0
	ldstr		"!!!! FAILURE !!!! -- LOCALLOC Tests FAIL -- !!!! FAILURE !!!!"



	callvirt	instance	void [mscorlib]System.IO.TextWriter::WriteLine(string)
	ldc.i4		0x0
	br			End

End:
	ret
.try TryStart to TryEnd filter Filter handler Handler to EndHandler
}
}
