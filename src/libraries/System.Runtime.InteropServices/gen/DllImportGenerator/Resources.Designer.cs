﻿//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//     Runtime Version:4.0.30319.42000
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace Microsoft.Interop {
    using System;
    
    
    /// <summary>
    ///   A strongly-typed resource class, for looking up localized strings, etc.
    /// </summary>
    // This class was auto-generated by the StronglyTypedResourceBuilder
    // class via a tool like ResGen or Visual Studio.
    // To add or remove a member, edit your .ResX file then rerun ResGen
    // with the /str option, or rebuild your VS project.
    [global::System.CodeDom.Compiler.GeneratedCodeAttribute("System.Resources.Tools.StronglyTypedResourceBuilder", "17.0.0.0")]
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
    [global::System.Runtime.CompilerServices.CompilerGeneratedAttribute()]
    internal class Resources {
        
        private static global::System.Resources.ResourceManager resourceMan;
        
        private static global::System.Globalization.CultureInfo resourceCulture;
        
        [global::System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1811:AvoidUncalledPrivateCode")]
        internal Resources() {
        }
        
        /// <summary>
        ///   Returns the cached ResourceManager instance used by this class.
        /// </summary>
        [global::System.ComponentModel.EditorBrowsableAttribute(global::System.ComponentModel.EditorBrowsableState.Advanced)]
        internal static global::System.Resources.ResourceManager ResourceManager {
            get {
                if (object.ReferenceEquals(resourceMan, null)) {
                    global::System.Resources.ResourceManager temp = new global::System.Resources.ResourceManager("Microsoft.Interop.Resources", typeof(Resources).Assembly);
                    resourceMan = temp;
                }
                return resourceMan;
            }
        }
        
        /// <summary>
        ///   Overrides the current thread's CurrentUICulture property for all
        ///   resource lookups using this strongly typed resource class.
        /// </summary>
        [global::System.ComponentModel.EditorBrowsableAttribute(global::System.ComponentModel.EditorBrowsableState.Advanced)]
        internal static global::System.Globalization.CultureInfo Culture {
            get {
                return resourceCulture;
            }
            set {
                resourceCulture = value;
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to A type marked with &apos;BlittableTypeAttribute&apos; must be blittable..
        /// </summary>
        internal static string BlittableTypeMustBeBlittableDescription {
            get {
                return ResourceManager.GetString("BlittableTypeMustBeBlittableDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Type &apos;{0}&apos; is marked with &apos;BlittableTypeAttribute&apos; but is not blittable.
        /// </summary>
        internal static string BlittableTypeMustBeBlittableMessage {
            get {
                return ResourceManager.GetString("BlittableTypeMustBeBlittableMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to When a constructor taking a Span&lt;byte&gt; is specified on the native type, the type must also have a public integer constant named BufferSize to provide the size of the caller-allocated buffer..
        /// </summary>
        internal static string CallerAllocConstructorMustHaveBufferSizeConstantDescription {
            get {
                return ResourceManager.GetString("CallerAllocConstructorMustHaveBufferSizeConstantDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type &apos;{0}&apos; must have a &apos;public const int BufferSize&apos; field that specifies the size of the stack buffer because it has a constructor that takes a caller-allocated Span&lt;byte&gt;.
        /// </summary>
        internal static string CallerAllocConstructorMustHaveBufferSizeConstantMessage {
            get {
                return ResourceManager.GetString("CallerAllocConstructorMustHaveBufferSizeConstantMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to A type that supports marshalling from managed to native using a caller-allocated buffer should also support marshalling from managed to native where using a caller-allocated buffer is impossible..
        /// </summary>
        internal static string CallerAllocMarshallingShouldSupportAllocatingMarshallingFallbackDescription {
            get {
                return ResourceManager.GetString("CallerAllocMarshallingShouldSupportAllocatingMarshallingFallbackDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Native type &apos;{0}&apos; has a constructor taking a caller-allocated buffer, but does not support marshalling in scenarios where using a caller-allocated buffer is impossible.
        /// </summary>
        internal static string CallerAllocMarshallingShouldSupportAllocatingMarshallingFallbackMessage {
            get {
                return ResourceManager.GetString("CallerAllocMarshallingShouldSupportAllocatingMarshallingFallbackMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The generated &apos;DllImportAttribute&apos; will not have a value corresponding to &apos;{0}&apos;..
        /// </summary>
        internal static string CannotForwardToDllImportDescription {
            get {
                return ResourceManager.GetString("CannotForwardToDllImportDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to &apos;{0}&apos; has no equivalent in &apos;DllImportAtttribute&apos; and will not be forwarded.
        /// </summary>
        internal static string CannotForwardToDllImportMessage {
            get {
                return ResourceManager.GetString("CannotForwardToDllImportMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Specified &apos;GeneratedDllImportAttribute&apos; arguments cannot be forwarded to &apos;DllImportAttribute&apos;.
        /// </summary>
        internal static string CannotForwardToDllImportTitle {
            get {
                return ResourceManager.GetString("CannotForwardToDllImportTitle", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The &apos;BlittableTypeAttribute&apos; and &apos;NativeMarshallingAttribute&apos; attributes are mutually exclusive..
        /// </summary>
        internal static string CannotHaveMultipleMarshallingAttributesDescription {
            get {
                return ResourceManager.GetString("CannotHaveMultipleMarshallingAttributesDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Type &apos;{0}&apos; is marked with &apos;BlittableTypeAttribute&apos; and &apos;NativeMarshallingAttribute&apos;. A type can only have one of these two attributes..
        /// </summary>
        internal static string CannotHaveMultipleMarshallingAttributesMessage {
            get {
                return ResourceManager.GetString("CannotHaveMultipleMarshallingAttributesMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to A native type with the &apos;GenericContiguousCollectionMarshallerAttribute&apos; must have at least one of the two marshalling methods as well as a &apos;ManagedValues&apos; property of type &apos;Span&lt;T&gt;&apos; for some &apos;T&apos; and a &apos;NativeValueStorage&apos; property of type &apos;Span&lt;byte&gt;&apos; to enable marshalling the managed type..
        /// </summary>
        internal static string CollectionNativeTypeMustHaveRequiredShapeDescription {
            get {
                return ResourceManager.GetString("CollectionNativeTypeMustHaveRequiredShapeDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type &apos;{0}&apos; must be a value type and have a constructor that takes two parameters, one of type &apos;{1}&apos; and an &apos;int&apos;, or have a parameterless instance method named &apos;ToManaged&apos; that returns &apos;{1}&apos; as well as a &apos;ManagedValues&apos; property of type &apos;Span&lt;T&gt;&apos; for some &apos;T&apos; and a &apos;NativeValueStorage&apos; property of type &apos;Span&lt;byte&gt;&apos;.
        /// </summary>
        internal static string CollectionNativeTypeMustHaveRequiredShapeMessage {
            get {
                return ResourceManager.GetString("CollectionNativeTypeMustHaveRequiredShapeMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Source-generated P/Invokes will ignore any configuration that is not supported..
        /// </summary>
        internal static string ConfigurationNotSupportedDescription {
            get {
                return ResourceManager.GetString("ConfigurationNotSupportedDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The &apos;{0}&apos; configuration is not supported by source-generated P/Invokes. If the specified configuration is required, use a regular `DllImport` instead..
        /// </summary>
        internal static string ConfigurationNotSupportedMessage {
            get {
                return ResourceManager.GetString("ConfigurationNotSupportedMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The specified marshalling configuration is not supported by source-generated P/Invokes. {0}..
        /// </summary>
        internal static string ConfigurationNotSupportedMessageMarshallingInfo {
            get {
                return ResourceManager.GetString("ConfigurationNotSupportedMessageMarshallingInfo", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The specified &apos;{0}&apos; configuration for parameter &apos;{1}&apos; is not supported by source-generated P/Invokes. If the specified configuration is required, use a regular `DllImport` instead..
        /// </summary>
        internal static string ConfigurationNotSupportedMessageParameter {
            get {
                return ResourceManager.GetString("ConfigurationNotSupportedMessageParameter", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The specified &apos;{0}&apos; configuration for the return value of method &apos;{1}&apos; is not supported by source-generated P/Invokes. If the specified configuration is required, use a regular `DllImport` instead..
        /// </summary>
        internal static string ConfigurationNotSupportedMessageReturn {
            get {
                return ResourceManager.GetString("ConfigurationNotSupportedMessageReturn", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The specified value &apos;{0}&apos; for &apos;{1}&apos; is not supported by source-generated P/Invokes. If the specified configuration is required, use a regular `DllImport` instead..
        /// </summary>
        internal static string ConfigurationNotSupportedMessageValue {
            get {
                return ResourceManager.GetString("ConfigurationNotSupportedMessageValue", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Specified configuration is not supported by source-generated P/Invokes..
        /// </summary>
        internal static string ConfigurationNotSupportedTitle {
            get {
                return ResourceManager.GetString("ConfigurationNotSupportedTitle", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Only one of &apos;ConstantElementCount&apos; or &apos;ElementCountInfo&apos; may be used in a &apos;MarshalUsingAttribute&apos; for a given &apos;ElementIndirectionLevel&apos;.
        /// </summary>
        internal static string ConstantAndElementCountInfoDisallowed {
            get {
                return ResourceManager.GetString("ConstantAndElementCountInfoDisallowed", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Automatically converting a P/Invoke with &apos;PreserveSig&apos; set to &apos;false&apos; to a source-generated P/Invoke may produce invalid code.
        /// </summary>
        internal static string ConvertNoPreserveSigDllImportToGeneratedMayProduceInvalidCode {
            get {
                return ResourceManager.GetString("ConvertNoPreserveSigDllImportToGeneratedMayProduceInvalidCode", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Convert to &apos;GeneratedDllImport&apos;.
        /// </summary>
        internal static string ConvertToGeneratedDllImport {
            get {
                return ResourceManager.GetString("ConvertToGeneratedDllImport", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Use &apos;GeneratedDllImportAttribute&apos; instead of &apos;DllImportAttribute&apos; to generate P/Invoke marshalling code at compile time.
        /// </summary>
        internal static string ConvertToGeneratedDllImportDescription {
            get {
                return ResourceManager.GetString("ConvertToGeneratedDllImportDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Mark the method &apos;{0}&apos; with &apos;GeneratedDllImportAttribute&apos; instead of &apos;DllImportAttribute&apos; to generate P/Invoke marshalling code at compile time.
        /// </summary>
        internal static string ConvertToGeneratedDllImportMessage {
            get {
                return ResourceManager.GetString("ConvertToGeneratedDllImportMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Use &apos;GeneratedDllImportAttribute&apos; instead of &apos;DllImportAttribute&apos; to generate P/Invoke marshalling code at compile time.
        /// </summary>
        internal static string ConvertToGeneratedDllImportTitle {
            get {
                return ResourceManager.GetString("ConvertToGeneratedDllImportTitle", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Conversion to &apos;GeneratedDllImport&apos; may change behavior and compatibility. See {0} for more information..
        /// </summary>
        internal static string ConvertToGeneratedDllImportWarning {
            get {
                return ResourceManager.GetString("ConvertToGeneratedDllImportWarning", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Convert to &apos;GeneratedDllImport&apos; with &apos;{0}&apos; suffix.
        /// </summary>
        internal static string ConvertToGeneratedDllImportWithSuffix {
            get {
                return ResourceManager.GetString("ConvertToGeneratedDllImportWithSuffix", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The specified parameter needs to be marshalled from managed to native, but the native type &apos;{0}&apos; does not support it..
        /// </summary>
        internal static string CustomTypeMarshallingManagedToNativeUnsupported {
            get {
                return ResourceManager.GetString("CustomTypeMarshallingManagedToNativeUnsupported", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The specified parameter needs to be marshalled from native to managed, but the native type &apos;{0}&apos; does not support it..
        /// </summary>
        internal static string CustomTypeMarshallingNativeToManagedUnsupported {
            get {
                return ResourceManager.GetString("CustomTypeMarshallingNativeToManagedUnsupported", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The return type of &apos;GetPinnableReference&apos; (after accounting for &apos;ref&apos;) must be blittable..
        /// </summary>
        internal static string GetPinnableReferenceReturnTypeBlittableDescription {
            get {
                return ResourceManager.GetString("GetPinnableReferenceReturnTypeBlittableDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The dereferenced type of the return type of the &apos;GetPinnableReference&apos; method must be blittable.
        /// </summary>
        internal static string GetPinnableReferenceReturnTypeBlittableMessage {
            get {
                return ResourceManager.GetString("GetPinnableReferenceReturnTypeBlittableMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to A type that supports marshalling from managed to native by pinning should also support marshalling from managed to native where pinning is impossible..
        /// </summary>
        internal static string GetPinnableReferenceShouldSupportAllocatingMarshallingFallbackDescription {
            get {
                return ResourceManager.GetString("GetPinnableReferenceShouldSupportAllocatingMarshallingFallbackDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Type &apos;{0}&apos; has a &apos;GetPinnableReference&apos; method but its native type does not support marshalling in scenarios where pinning is impossible.
        /// </summary>
        internal static string GetPinnableReferenceShouldSupportAllocatingMarshallingFallbackMessage {
            get {
                return ResourceManager.GetString("GetPinnableReferenceShouldSupportAllocatingMarshallingFallbackMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Method &apos;{0}&apos; is contained in a type &apos;{1}&apos; that is not marked &apos;partial&apos;. P/Invoke source generation will ignore method &apos;{0}&apos;..
        /// </summary>
        internal static string InvalidAttributedMethodContainingTypeMissingModifiersMessage {
            get {
                return ResourceManager.GetString("InvalidAttributedMethodContainingTypeMissingModifiersMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Methods marked with &apos;GeneratedDllImportAttribute&apos; should be &apos;static&apos;, &apos;partial&apos;, and non-generic. P/Invoke source generation will ignore methods that are non-&apos;static&apos;, non-&apos;partial&apos;, or generic..
        /// </summary>
        internal static string InvalidAttributedMethodDescription {
            get {
                return ResourceManager.GetString("InvalidAttributedMethodDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Method &apos;{0}&apos; should be &apos;static&apos;, &apos;partial&apos;, and non-generic when marked with &apos;GeneratedDllImportAttribute&apos;. P/Invoke source generation will ignore method &apos;{0}&apos;..
        /// </summary>
        internal static string InvalidAttributedMethodSignatureMessage {
            get {
                return ResourceManager.GetString("InvalidAttributedMethodSignatureMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Invalid &apos;LibraryImportAttribute&apos; usage.
        /// </summary>
        internal static string InvalidLibraryImportAttributeUsageTitle {
            get {
                return ResourceManager.GetString("InvalidLibraryImportAttributeUsageTitle", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The configuration of &apos;StringMarshalling&apos; and &apos;StringMarshallingCustomType&apos; is invalid..
        /// </summary>
        internal static string InvalidStringMarshallingConfigurationDescription {
            get {
                return ResourceManager.GetString("InvalidStringMarshallingConfigurationDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The configuration of &apos;StringMarshalling&apos; and &apos;StringMarshallingCustomType&apos; on method &apos;{0}&apos; is invalid. {1}.
        /// </summary>
        internal static string InvalidStringMarshallingConfigurationMessage {
            get {
                return ResourceManager.GetString("InvalidStringMarshallingConfigurationMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to &apos;StringMarshallingCustomType&apos; must be specified when &apos;StringMarshalling&apos; is set to &apos;StringMarshalling.Custom&apos;..
        /// </summary>
        internal static string InvalidStringMarshallingConfigurationMissingCustomType {
            get {
                return ResourceManager.GetString("InvalidStringMarshallingConfigurationMissingCustomType", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to &apos;StringMarshalling&apos; should be set to &apos;StringMarshalling.Custom&apos; when &apos;StringMarshallingCustomType&apos; is specified..
        /// </summary>
        internal static string InvalidStringMarshallingConfigurationNotCustom {
            get {
                return ResourceManager.GetString("InvalidStringMarshallingConfigurationNotCustom", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The use cases for &apos;GetPinnableReference&apos; are not applicable in any scenarios where a &apos;Value&apos; property is not also required..
        /// </summary>
        internal static string MarshallerGetPinnableReferenceRequiresValuePropertyDescription {
            get {
                return ResourceManager.GetString("MarshallerGetPinnableReferenceRequiresValuePropertyDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The &apos;GetPinnableReference&apos; method cannot be provided on the native type &apos;{0}&apos; unless a &apos;Value&apos; property is also provided.
        /// </summary>
        internal static string MarshallerGetPinnableReferenceRequiresValuePropertyMessage {
            get {
                return ResourceManager.GetString("MarshallerGetPinnableReferenceRequiresValuePropertyMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type &apos;{0}&apos; must be a closed generic so the emitted code can use a specific instantiation..
        /// </summary>
        internal static string NativeGenericTypeMustBeClosedDescription {
            get {
                return ResourceManager.GetString("NativeGenericTypeMustBeClosedDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type &apos;{0}&apos; must be a closed generic or have the same number of generic parameters as the managed type so the emitted code can use a specific instantiation..
        /// </summary>
        internal static string NativeGenericTypeMustBeClosedOrMatchArityDescription {
            get {
                return ResourceManager.GetString("NativeGenericTypeMustBeClosedOrMatchArityDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type &apos;{0}&apos; for managed type &apos;{1}&apos; must be a closed generic type or have the same arity as the managed type..
        /// </summary>
        internal static string NativeGenericTypeMustBeClosedOrMatchArityMessage {
            get {
                return ResourceManager.GetString("NativeGenericTypeMustBeClosedOrMatchArityMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to A native type for a given type must be blittable..
        /// </summary>
        internal static string NativeTypeMustBeBlittableDescription {
            get {
                return ResourceManager.GetString("NativeTypeMustBeBlittableDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type &apos;{0}&apos; for the type &apos;{1}&apos; is not blittable.
        /// </summary>
        internal static string NativeTypeMustBeBlittableMessage {
            get {
                return ResourceManager.GetString("NativeTypeMustBeBlittableMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to A native type for a given type must be non-null..
        /// </summary>
        internal static string NativeTypeMustBeNonNullDescription {
            get {
                return ResourceManager.GetString("NativeTypeMustBeNonNullDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type for the type &apos;{0}&apos; is null.
        /// </summary>
        internal static string NativeTypeMustBeNonNullMessage {
            get {
                return ResourceManager.GetString("NativeTypeMustBeNonNullMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type must be pointer sized so the pinned result of &apos;GetPinnableReference&apos; can be cast to the native type..
        /// </summary>
        internal static string NativeTypeMustBePointerSizedDescription {
            get {
                return ResourceManager.GetString("NativeTypeMustBePointerSizedDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type &apos;{0}&apos; must be pointer sized because the managed type &apos;{1}&apos; has a &apos;GetPinnableReference&apos; method.
        /// </summary>
        internal static string NativeTypeMustBePointerSizedMessage {
            get {
                return ResourceManager.GetString("NativeTypeMustBePointerSizedMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type must have at least one of the two marshalling methods to enable marshalling the managed type..
        /// </summary>
        internal static string NativeTypeMustHaveRequiredShapeDescription {
            get {
                return ResourceManager.GetString("NativeTypeMustHaveRequiredShapeDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type &apos;{0}&apos; must be a value type and have a constructor that takes one parameter of type &apos;{1}&apos; or a parameterless instance method named &apos;ToManaged&apos; that returns &apos;{1}&apos;.
        /// </summary>
        internal static string NativeTypeMustHaveRequiredShapeMessage {
            get {
                return ResourceManager.GetString("NativeTypeMustHaveRequiredShapeMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The &apos;Value&apos; property must not be a &apos;ref&apos; or &apos;readonly ref&apos; property..
        /// </summary>
        internal static string RefValuePropertyUnsupportedDescription {
            get {
                return ResourceManager.GetString("RefValuePropertyUnsupportedDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The &apos;Value&apos; property on the native type &apos;{0}&apos; must not be a &apos;ref&apos; or &apos;readonly ref&apos; property..
        /// </summary>
        internal static string RefValuePropertyUnsupportedMessage {
            get {
                return ResourceManager.GetString("RefValuePropertyUnsupportedMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to .
        /// </summary>
        internal static string RuntimeMarshallingMustBeDisabled {
            get {
                return ResourceManager.GetString("RuntimeMarshallingMustBeDisabled", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to An abstract type derived from &apos;SafeHandle&apos; cannot be marshalled by reference. The provided type must be concrete..
        /// </summary>
        internal static string SafeHandleByRefMustBeConcrete {
            get {
                return ResourceManager.GetString("SafeHandleByRefMustBeConcrete", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to P/Invoke source generation is not supported on unknown target framework v{0}. The generated source will not be compatible with other frameworks..
        /// </summary>
        internal static string TargetFrameworkNotSupportedDescription {
            get {
                return ResourceManager.GetString("TargetFrameworkNotSupportedDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to &apos;GeneratedDllImportAttribute&apos; cannot be used for source-generated P/Invokes on an unknown target framework v{0}..
        /// </summary>
        internal static string TargetFrameworkNotSupportedMessage {
            get {
                return ResourceManager.GetString("TargetFrameworkNotSupportedMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Current target framework is not supported by source-generated P/Invokes.
        /// </summary>
        internal static string TargetFrameworkNotSupportedTitle {
            get {
                return ResourceManager.GetString("TargetFrameworkNotSupportedTitle", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to For types that are not supported by source-generated P/Invokes, the resulting P/Invoke will rely on the underlying runtime to marshal the specified type..
        /// </summary>
        internal static string TypeNotSupportedDescription {
            get {
                return ResourceManager.GetString("TypeNotSupportedDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The type &apos;{0}&apos; is not supported by source-generated P/Invokes. The generated source will not handle marshalling of parameter &apos;{1}&apos;..
        /// </summary>
        internal static string TypeNotSupportedMessageParameter {
            get {
                return ResourceManager.GetString("TypeNotSupportedMessageParameter", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to {0} The generated source will not handle marshalling of parameter &apos;{1}&apos;..
        /// </summary>
        internal static string TypeNotSupportedMessageParameterWithDetails {
            get {
                return ResourceManager.GetString("TypeNotSupportedMessageParameterWithDetails", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The type &apos;{0}&apos; is not supported by source-generated P/Invokes. The generated source will not handle marshalling of the return value of method &apos;{1}&apos;..
        /// </summary>
        internal static string TypeNotSupportedMessageReturn {
            get {
                return ResourceManager.GetString("TypeNotSupportedMessageReturn", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to {0} The generated source will not handle marshalling of the return value of method &apos;{1}&apos;..
        /// </summary>
        internal static string TypeNotSupportedMessageReturnWithDetails {
            get {
                return ResourceManager.GetString("TypeNotSupportedMessageReturnWithDetails", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Specified type is not supported by source-generated P/Invokes.
        /// </summary>
        internal static string TypeNotSupportedTitle {
            get {
                return ResourceManager.GetString("TypeNotSupportedTitle", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type&apos;s &apos;Value&apos; property must have a getter to support marshalling from managed to native..
        /// </summary>
        internal static string ValuePropertyMustHaveGetterDescription {
            get {
                return ResourceManager.GetString("ValuePropertyMustHaveGetterDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The &apos;Value&apos; property on the native type &apos;{0}&apos; must have a getter.
        /// </summary>
        internal static string ValuePropertyMustHaveGetterMessage {
            get {
                return ResourceManager.GetString("ValuePropertyMustHaveGetterMessage", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The native type&apos;s &apos;Value&apos; property must have a setter to support marshalling from native to managed..
        /// </summary>
        internal static string ValuePropertyMustHaveSetterDescription {
            get {
                return ResourceManager.GetString("ValuePropertyMustHaveSetterDescription", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to The &apos;Value&apos; property on the native type &apos;{0}&apos; must have a setter.
        /// </summary>
        internal static string ValuePropertyMustHaveSetterMessage {
            get {
                return ResourceManager.GetString("ValuePropertyMustHaveSetterMessage", resourceCulture);
            }
        }
    }
}
