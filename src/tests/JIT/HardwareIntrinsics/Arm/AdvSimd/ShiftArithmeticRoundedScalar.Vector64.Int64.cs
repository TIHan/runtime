// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

/******************************************************************************
 * This file is auto-generated from a template file by the GenerateTests.csx  *
 * script in tests\src\JIT\HardwareIntrinsics.Arm\Shared. In order to make    *
 * changes, please update the corresponding template and run according to the *
 * directions listed in the file.                                             *
 ******************************************************************************/

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using Xunit;

namespace JIT.HardwareIntrinsics.Arm._AdvSimd
{
    public static partial class Program
    {
        [Fact]
        public static void ShiftArithmeticRoundedScalar_Vector64_Int64()
        {
            var test = new SimpleBinaryOpTest__ShiftArithmeticRoundedScalar_Vector64_Int64();

            if (test.IsSupported)
            {
                // Validates basic functionality works, using Unsafe.Read
                test.RunBasicScenario_UnsafeRead();

                if (AdvSimd.IsSupported)
                {
                    // Validates basic functionality works, using Load
                    test.RunBasicScenario_Load();
                }

                // Validates calling via reflection works, using Unsafe.Read
                test.RunReflectionScenario_UnsafeRead();

                if (AdvSimd.IsSupported)
                {
                    // Validates calling via reflection works, using Load
                    test.RunReflectionScenario_Load();
                }

                // Validates passing a static member works
                test.RunClsVarScenario();

                if (AdvSimd.IsSupported)
                {
                    // Validates passing a static member works, using pinning and Load
                    test.RunClsVarScenario_Load();
                }

                // Validates passing a local works, using Unsafe.Read
                test.RunLclVarScenario_UnsafeRead();

                if (AdvSimd.IsSupported)
                {
                    // Validates passing a local works, using Load
                    test.RunLclVarScenario_Load();
                }

                // Validates passing the field of a local class works
                test.RunClassLclFldScenario();

                if (AdvSimd.IsSupported)
                {
                    // Validates passing the field of a local class works, using pinning and Load
                    test.RunClassLclFldScenario_Load();
                }

                // Validates passing an instance member of a class works
                test.RunClassFldScenario();

                if (AdvSimd.IsSupported)
                {
                    // Validates passing an instance member of a class works, using pinning and Load
                    test.RunClassFldScenario_Load();
                }

                // Validates passing the field of a local struct works
                test.RunStructLclFldScenario();

                if (AdvSimd.IsSupported)
                {
                    // Validates passing the field of a local struct works, using pinning and Load
                    test.RunStructLclFldScenario_Load();
                }

                // Validates passing an instance member of a struct works
                test.RunStructFldScenario();

                if (AdvSimd.IsSupported)
                {
                    // Validates passing an instance member of a struct works, using pinning and Load
                    test.RunStructFldScenario_Load();
                }
            }
            else
            {
                // Validates we throw on unsupported hardware
                test.RunUnsupportedScenario();
            }

            if (!test.Succeeded)
            {
                throw new Exception("One or more scenarios did not complete as expected.");
            }
        }
    }

    public sealed unsafe class SimpleBinaryOpTest__ShiftArithmeticRoundedScalar_Vector64_Int64
    {
        private struct DataTable
        {
            private byte[] inArray1;
            private byte[] inArray2;
            private byte[] outArray;

            private GCHandle inHandle1;
            private GCHandle inHandle2;
            private GCHandle outHandle;

            private ulong alignment;

            public DataTable(Int64[] inArray1, Int64[] inArray2, Int64[] outArray, int alignment)
            {
                int sizeOfinArray1 = inArray1.Length * Unsafe.SizeOf<Int64>();
                int sizeOfinArray2 = inArray2.Length * Unsafe.SizeOf<Int64>();
                int sizeOfoutArray = outArray.Length * Unsafe.SizeOf<Int64>();
                if ((alignment != 16 && alignment != 8) || (alignment * 2) < sizeOfinArray1 || (alignment * 2) < sizeOfinArray2 || (alignment * 2) < sizeOfoutArray)
                {
                    throw new ArgumentException("Invalid value of alignment");
                }

                this.inArray1 = new byte[alignment * 2];
                this.inArray2 = new byte[alignment * 2];
                this.outArray = new byte[alignment * 2];

                this.inHandle1 = GCHandle.Alloc(this.inArray1, GCHandleType.Pinned);
                this.inHandle2 = GCHandle.Alloc(this.inArray2, GCHandleType.Pinned);
                this.outHandle = GCHandle.Alloc(this.outArray, GCHandleType.Pinned);

                this.alignment = (ulong)alignment;

                Unsafe.CopyBlockUnaligned(ref Unsafe.AsRef<byte>(inArray1Ptr), ref Unsafe.As<Int64, byte>(ref inArray1[0]), (uint)sizeOfinArray1);
                Unsafe.CopyBlockUnaligned(ref Unsafe.AsRef<byte>(inArray2Ptr), ref Unsafe.As<Int64, byte>(ref inArray2[0]), (uint)sizeOfinArray2);
            }

            public void* inArray1Ptr => Align((byte*)(inHandle1.AddrOfPinnedObject().ToPointer()), alignment);
            public void* inArray2Ptr => Align((byte*)(inHandle2.AddrOfPinnedObject().ToPointer()), alignment);
            public void* outArrayPtr => Align((byte*)(outHandle.AddrOfPinnedObject().ToPointer()), alignment);

            public void Dispose()
            {
                inHandle1.Free();
                inHandle2.Free();
                outHandle.Free();
            }

            private static unsafe void* Align(byte* buffer, ulong expectedAlignment)
            {
                return (void*)(((ulong)buffer + expectedAlignment - 1) & ~(expectedAlignment - 1));
            }
        }

        private struct TestStruct
        {
            public Vector64<Int64> _fld1;
            public Vector64<Int64> _fld2;

            public static TestStruct Create()
            {
                var testStruct = new TestStruct();

                for (var i = 0; i < Op1ElementCount; i++) { _data1[i] = TestLibrary.Generator.GetInt64(); }
                Unsafe.CopyBlockUnaligned(ref Unsafe.As<Vector64<Int64>, byte>(ref testStruct._fld1), ref Unsafe.As<Int64, byte>(ref _data1[0]), (uint)Unsafe.SizeOf<Vector64<Int64>>());
                for (var i = 0; i < Op2ElementCount; i++) { _data2[i] = TestLibrary.Generator.GetInt64(); }
                Unsafe.CopyBlockUnaligned(ref Unsafe.As<Vector64<Int64>, byte>(ref testStruct._fld2), ref Unsafe.As<Int64, byte>(ref _data2[0]), (uint)Unsafe.SizeOf<Vector64<Int64>>());

                return testStruct;
            }

            public void RunStructFldScenario(SimpleBinaryOpTest__ShiftArithmeticRoundedScalar_Vector64_Int64 testClass)
            {
                var result = AdvSimd.ShiftArithmeticRoundedScalar(_fld1, _fld2);

                Unsafe.Write(testClass._dataTable.outArrayPtr, result);
                testClass.ValidateResult(_fld1, _fld2, testClass._dataTable.outArrayPtr);
            }

            public void RunStructFldScenario_Load(SimpleBinaryOpTest__ShiftArithmeticRoundedScalar_Vector64_Int64 testClass)
            {
                fixed (Vector64<Int64>* pFld1 = &_fld1)
                fixed (Vector64<Int64>* pFld2 = &_fld2)
                {
                    var result = AdvSimd.ShiftArithmeticRoundedScalar(
                        AdvSimd.LoadVector64((Int64*)(pFld1)),
                        AdvSimd.LoadVector64((Int64*)(pFld2))
                    );

                    Unsafe.Write(testClass._dataTable.outArrayPtr, result);
                    testClass.ValidateResult(_fld1, _fld2, testClass._dataTable.outArrayPtr);
                }
            }
        }

        private static readonly int LargestVectorSize = 8;

        private static readonly int Op1ElementCount = Unsafe.SizeOf<Vector64<Int64>>() / sizeof(Int64);
        private static readonly int Op2ElementCount = Unsafe.SizeOf<Vector64<Int64>>() / sizeof(Int64);
        private static readonly int RetElementCount = Unsafe.SizeOf<Vector64<Int64>>() / sizeof(Int64);

        private static Int64[] _data1 = new Int64[Op1ElementCount];
        private static Int64[] _data2 = new Int64[Op2ElementCount];

        private static Vector64<Int64> _clsVar1;
        private static Vector64<Int64> _clsVar2;

        private Vector64<Int64> _fld1;
        private Vector64<Int64> _fld2;

        private DataTable _dataTable;

        static SimpleBinaryOpTest__ShiftArithmeticRoundedScalar_Vector64_Int64()
        {
            for (var i = 0; i < Op1ElementCount; i++) { _data1[i] = TestLibrary.Generator.GetInt64(); }
            Unsafe.CopyBlockUnaligned(ref Unsafe.As<Vector64<Int64>, byte>(ref _clsVar1), ref Unsafe.As<Int64, byte>(ref _data1[0]), (uint)Unsafe.SizeOf<Vector64<Int64>>());
            for (var i = 0; i < Op2ElementCount; i++) { _data2[i] = TestLibrary.Generator.GetInt64(); }
            Unsafe.CopyBlockUnaligned(ref Unsafe.As<Vector64<Int64>, byte>(ref _clsVar2), ref Unsafe.As<Int64, byte>(ref _data2[0]), (uint)Unsafe.SizeOf<Vector64<Int64>>());
        }

        public SimpleBinaryOpTest__ShiftArithmeticRoundedScalar_Vector64_Int64()
        {
            Succeeded = true;

            for (var i = 0; i < Op1ElementCount; i++) { _data1[i] = TestLibrary.Generator.GetInt64(); }
            Unsafe.CopyBlockUnaligned(ref Unsafe.As<Vector64<Int64>, byte>(ref _fld1), ref Unsafe.As<Int64, byte>(ref _data1[0]), (uint)Unsafe.SizeOf<Vector64<Int64>>());
            for (var i = 0; i < Op2ElementCount; i++) { _data2[i] = TestLibrary.Generator.GetInt64(); }
            Unsafe.CopyBlockUnaligned(ref Unsafe.As<Vector64<Int64>, byte>(ref _fld2), ref Unsafe.As<Int64, byte>(ref _data2[0]), (uint)Unsafe.SizeOf<Vector64<Int64>>());

            for (var i = 0; i < Op1ElementCount; i++) { _data1[i] = TestLibrary.Generator.GetInt64(); }
            for (var i = 0; i < Op2ElementCount; i++) { _data2[i] = TestLibrary.Generator.GetInt64(); }
            _dataTable = new DataTable(_data1, _data2, new Int64[RetElementCount], LargestVectorSize);
        }

        public bool IsSupported => AdvSimd.IsSupported;

        public bool Succeeded { get; set; }

        public void RunBasicScenario_UnsafeRead()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunBasicScenario_UnsafeRead));

            var result = AdvSimd.ShiftArithmeticRoundedScalar(
                Unsafe.Read<Vector64<Int64>>(_dataTable.inArray1Ptr),
                Unsafe.Read<Vector64<Int64>>(_dataTable.inArray2Ptr)
            );

            Unsafe.Write(_dataTable.outArrayPtr, result);
            ValidateResult(_dataTable.inArray1Ptr, _dataTable.inArray2Ptr, _dataTable.outArrayPtr);
        }

        public void RunBasicScenario_Load()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunBasicScenario_Load));

            var result = AdvSimd.ShiftArithmeticRoundedScalar(
                AdvSimd.LoadVector64((Int64*)(_dataTable.inArray1Ptr)),
                AdvSimd.LoadVector64((Int64*)(_dataTable.inArray2Ptr))
            );

            Unsafe.Write(_dataTable.outArrayPtr, result);
            ValidateResult(_dataTable.inArray1Ptr, _dataTable.inArray2Ptr, _dataTable.outArrayPtr);
        }

        public void RunReflectionScenario_UnsafeRead()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunReflectionScenario_UnsafeRead));

            var result = typeof(AdvSimd).GetMethod(nameof(AdvSimd.ShiftArithmeticRoundedScalar), new Type[] { typeof(Vector64<Int64>), typeof(Vector64<Int64>) })
                                     .Invoke(null, new object[] {
                                        Unsafe.Read<Vector64<Int64>>(_dataTable.inArray1Ptr),
                                        Unsafe.Read<Vector64<Int64>>(_dataTable.inArray2Ptr)
                                     });

            Unsafe.Write(_dataTable.outArrayPtr, (Vector64<Int64>)(result));
            ValidateResult(_dataTable.inArray1Ptr, _dataTable.inArray2Ptr, _dataTable.outArrayPtr);
        }

        public void RunReflectionScenario_Load()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunReflectionScenario_Load));

            var result = typeof(AdvSimd).GetMethod(nameof(AdvSimd.ShiftArithmeticRoundedScalar), new Type[] { typeof(Vector64<Int64>), typeof(Vector64<Int64>) })
                                     .Invoke(null, new object[] {
                                        AdvSimd.LoadVector64((Int64*)(_dataTable.inArray1Ptr)),
                                        AdvSimd.LoadVector64((Int64*)(_dataTable.inArray2Ptr))
                                     });

            Unsafe.Write(_dataTable.outArrayPtr, (Vector64<Int64>)(result));
            ValidateResult(_dataTable.inArray1Ptr, _dataTable.inArray2Ptr, _dataTable.outArrayPtr);
        }

        public void RunClsVarScenario()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunClsVarScenario));

            var result = AdvSimd.ShiftArithmeticRoundedScalar(
                _clsVar1,
                _clsVar2
            );

            Unsafe.Write(_dataTable.outArrayPtr, result);
            ValidateResult(_clsVar1, _clsVar2, _dataTable.outArrayPtr);
        }

        public void RunClsVarScenario_Load()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunClsVarScenario_Load));

            fixed (Vector64<Int64>* pClsVar1 = &_clsVar1)
            fixed (Vector64<Int64>* pClsVar2 = &_clsVar2)
            {
                var result = AdvSimd.ShiftArithmeticRoundedScalar(
                    AdvSimd.LoadVector64((Int64*)(pClsVar1)),
                    AdvSimd.LoadVector64((Int64*)(pClsVar2))
                );

                Unsafe.Write(_dataTable.outArrayPtr, result);
                ValidateResult(_clsVar1, _clsVar2, _dataTable.outArrayPtr);
            }
        }

        public void RunLclVarScenario_UnsafeRead()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunLclVarScenario_UnsafeRead));

            var op1 = Unsafe.Read<Vector64<Int64>>(_dataTable.inArray1Ptr);
            var op2 = Unsafe.Read<Vector64<Int64>>(_dataTable.inArray2Ptr);
            var result = AdvSimd.ShiftArithmeticRoundedScalar(op1, op2);

            Unsafe.Write(_dataTable.outArrayPtr, result);
            ValidateResult(op1, op2, _dataTable.outArrayPtr);
        }

        public void RunLclVarScenario_Load()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunLclVarScenario_Load));

            var op1 = AdvSimd.LoadVector64((Int64*)(_dataTable.inArray1Ptr));
            var op2 = AdvSimd.LoadVector64((Int64*)(_dataTable.inArray2Ptr));
            var result = AdvSimd.ShiftArithmeticRoundedScalar(op1, op2);

            Unsafe.Write(_dataTable.outArrayPtr, result);
            ValidateResult(op1, op2, _dataTable.outArrayPtr);
        }

        public void RunClassLclFldScenario()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunClassLclFldScenario));

            var test = new SimpleBinaryOpTest__ShiftArithmeticRoundedScalar_Vector64_Int64();
            var result = AdvSimd.ShiftArithmeticRoundedScalar(test._fld1, test._fld2);

            Unsafe.Write(_dataTable.outArrayPtr, result);
            ValidateResult(test._fld1, test._fld2, _dataTable.outArrayPtr);
        }

        public void RunClassLclFldScenario_Load()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunClassLclFldScenario_Load));

            var test = new SimpleBinaryOpTest__ShiftArithmeticRoundedScalar_Vector64_Int64();

            fixed (Vector64<Int64>* pFld1 = &test._fld1)
            fixed (Vector64<Int64>* pFld2 = &test._fld2)
            {
                var result = AdvSimd.ShiftArithmeticRoundedScalar(
                    AdvSimd.LoadVector64((Int64*)(pFld1)),
                    AdvSimd.LoadVector64((Int64*)(pFld2))
                );

                Unsafe.Write(_dataTable.outArrayPtr, result);
                ValidateResult(test._fld1, test._fld2, _dataTable.outArrayPtr);
            }
        }

        public void RunClassFldScenario()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunClassFldScenario));

            var result = AdvSimd.ShiftArithmeticRoundedScalar(_fld1, _fld2);

            Unsafe.Write(_dataTable.outArrayPtr, result);
            ValidateResult(_fld1, _fld2, _dataTable.outArrayPtr);
        }

        public void RunClassFldScenario_Load()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunClassFldScenario_Load));

            fixed (Vector64<Int64>* pFld1 = &_fld1)
            fixed (Vector64<Int64>* pFld2 = &_fld2)
            {
                var result = AdvSimd.ShiftArithmeticRoundedScalar(
                    AdvSimd.LoadVector64((Int64*)(pFld1)),
                    AdvSimd.LoadVector64((Int64*)(pFld2))
                );

                Unsafe.Write(_dataTable.outArrayPtr, result);
                ValidateResult(_fld1, _fld2, _dataTable.outArrayPtr);
            }
        }

        public void RunStructLclFldScenario()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunStructLclFldScenario));

            var test = TestStruct.Create();
            var result = AdvSimd.ShiftArithmeticRoundedScalar(test._fld1, test._fld2);

            Unsafe.Write(_dataTable.outArrayPtr, result);
            ValidateResult(test._fld1, test._fld2, _dataTable.outArrayPtr);
        }

        public void RunStructLclFldScenario_Load()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunStructLclFldScenario_Load));

            var test = TestStruct.Create();
            var result = AdvSimd.ShiftArithmeticRoundedScalar(
                AdvSimd.LoadVector64((Int64*)(&test._fld1)),
                AdvSimd.LoadVector64((Int64*)(&test._fld2))
            );

            Unsafe.Write(_dataTable.outArrayPtr, result);
            ValidateResult(test._fld1, test._fld2, _dataTable.outArrayPtr);
        }

        public void RunStructFldScenario()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunStructFldScenario));

            var test = TestStruct.Create();
            test.RunStructFldScenario(this);
        }

        public void RunStructFldScenario_Load()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunStructFldScenario_Load));

            var test = TestStruct.Create();
            test.RunStructFldScenario_Load(this);
        }

        public void RunUnsupportedScenario()
        {
            TestLibrary.TestFramework.BeginScenario(nameof(RunUnsupportedScenario));

            bool succeeded = false;

            try
            {
                RunBasicScenario_UnsafeRead();
            }
            catch (PlatformNotSupportedException)
            {
                succeeded = true;
            }

            if (!succeeded)
            {
                Succeeded = false;
            }
        }

        private void ValidateResult(Vector64<Int64> op1, Vector64<Int64> op2, void* result, [CallerMemberName] string method = "")
        {
            Int64[] inArray1 = new Int64[Op1ElementCount];
            Int64[] inArray2 = new Int64[Op2ElementCount];
            Int64[] outArray = new Int64[RetElementCount];

            Unsafe.WriteUnaligned(ref Unsafe.As<Int64, byte>(ref inArray1[0]), op1);
            Unsafe.WriteUnaligned(ref Unsafe.As<Int64, byte>(ref inArray2[0]), op2);
            Unsafe.CopyBlockUnaligned(ref Unsafe.As<Int64, byte>(ref outArray[0]), ref Unsafe.AsRef<byte>(result), (uint)Unsafe.SizeOf<Vector64<Int64>>());

            ValidateResult(inArray1, inArray2, outArray, method);
        }

        private void ValidateResult(void* op1, void* op2, void* result, [CallerMemberName] string method = "")
        {
            Int64[] inArray1 = new Int64[Op1ElementCount];
            Int64[] inArray2 = new Int64[Op2ElementCount];
            Int64[] outArray = new Int64[RetElementCount];

            Unsafe.CopyBlockUnaligned(ref Unsafe.As<Int64, byte>(ref inArray1[0]), ref Unsafe.AsRef<byte>(op1), (uint)Unsafe.SizeOf<Vector64<Int64>>());
            Unsafe.CopyBlockUnaligned(ref Unsafe.As<Int64, byte>(ref inArray2[0]), ref Unsafe.AsRef<byte>(op2), (uint)Unsafe.SizeOf<Vector64<Int64>>());
            Unsafe.CopyBlockUnaligned(ref Unsafe.As<Int64, byte>(ref outArray[0]), ref Unsafe.AsRef<byte>(result), (uint)Unsafe.SizeOf<Vector64<Int64>>());

            ValidateResult(inArray1, inArray2, outArray, method);
        }

        private void ValidateResult(Int64[] left, Int64[] right, Int64[] result, [CallerMemberName] string method = "")
        {
            bool succeeded = true;

            if (Helpers.ShiftArithmeticRounded(left[0], right[0]) != result[0])
            {
                succeeded = false;
            }
            else
            {
                for (var i = 1; i < RetElementCount; i++)
                {
                    if (result[i] != 0)
                    {
                        succeeded = false;
                        break;
                    }
                }
            }

            if (!succeeded)
            {
                TestLibrary.TestFramework.LogInformation($"{nameof(AdvSimd)}.{nameof(AdvSimd.ShiftArithmeticRoundedScalar)}<Int64>(Vector64<Int64>, Vector64<Int64>): {method} failed:");
                TestLibrary.TestFramework.LogInformation($"    left: ({string.Join(", ", left)})");
                TestLibrary.TestFramework.LogInformation($"   right: ({string.Join(", ", right)})");
                TestLibrary.TestFramework.LogInformation($"  result: ({string.Join(", ", result)})");
                TestLibrary.TestFramework.LogInformation(string.Empty);

                Succeeded = false;
            }
        }
    }
}
