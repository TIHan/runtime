// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Runtime.CompilerServices;

public class Program
{
    public static int Main()
    {
        byte vr4 = default(byte);
        var vr5 = (byte)~vr4;
        uint vr6 = (uint)M50(vr5);

        if (vr6 != 255)
        {
            throw new Exception("Failed");
        }

        System.Console.WriteLine(vr6);
        Test();

        return 100;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void Test()
    {
        M1(0);
    }

    public static void M1(byte arg0)
    {
        long var6 = default(long);
        arg0 = (byte)(~(ulong)var6 % 3545460779U);
        System.Console.WriteLine(arg0);
    }

    public static long M50(byte arg0)
    {
        arg0 = 246;
        arg0 = (byte)(-1 % (arg0 | 1));
        return arg0;
    }
}
