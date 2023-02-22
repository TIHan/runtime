// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

// Generated by Fuzzlyn v1.5 on 2023-02-22 22:21:43
// Run on X64 Windows
// Seed: 1391301158118545956
public class Program
{
    public static IRuntime s_rt;
    public static ulong s_2 = 12180724836008689112UL;
    public static int Main()
    {
        s_rt = new Runtime();
        short vr4 = default(short);
        long vr5 = M3(vr4);
        if (vr5 == 4294967295)
        {
            return 100;
        }
        return 0;
    }

    public static long M3(short arg0)
    {
        s_2 = (uint)(-1L % (-(int)s_2));
        s_rt.WriteLine(arg0);
        return arg0;
    }
}

public interface IRuntime
{
    void WriteLine<T>(T value);
}

public class Runtime : IRuntime
{
    public void WriteLine<T>(T value) => System.Console.WriteLine(value);
}