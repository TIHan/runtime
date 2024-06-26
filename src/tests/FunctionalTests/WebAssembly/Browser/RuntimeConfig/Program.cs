// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices.JavaScript;

namespace Sample
{
    public partial class Test
    {
        public static void Main(string[] args)
        {
            Console.WriteLine ("Hello, World!");
        }

        [JSExport]
        public static int TestMeaning()
        {
            var config = AppContext.GetData("test_runtimeconfig_json");
            int result = ((string)config).Equals("25") ? 42 : 1;
            return result;
        }
    }
}