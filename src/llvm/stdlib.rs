use std::ffi::{c_char, c_void, CStr};

use const_format::concatcp;

use crate::parser::{SimpleType, Type};

pub const FN_NAME_PREFIX: &str = "mila_";

pub struct ExternalProcedure<'a> {
    pub c_name: &'a CStr,
    pub c_fn_ptr: *mut c_void,
    pub name: &'a str,
    pub param_type: Type,
    pub takes_string: bool,
}

// TODO this will probably work even with reference functions
// as long as Mila runs single-threaded
unsafe impl<'a> Sync for ExternalProcedure<'a> {}

macro_rules! cstr {
    ($str:expr) => {
        unsafe { CStr::from_bytes_with_nul_unchecked(concatcp!($str, "\0").as_bytes()) }
    };
}

#[used]
pub static EXTERNAL_PROCS: [ExternalProcedure; 6] = [
    ExternalProcedure {
        c_name: cstr!(concatcp!(FN_NAME_PREFIX, "write")),
        c_fn_ptr: mila_write as *mut c_void,
        name: "write",
        param_type: Type::Simple(SimpleType::Integer),
        takes_string: false,
    },
    ExternalProcedure {
        c_name: cstr!(concatcp!(FN_NAME_PREFIX, "writeln")),
        c_fn_ptr: mila_writeln as *mut c_void,
        name: "writeln",
        param_type: Type::Simple(SimpleType::Integer),
        takes_string: false,
    },
    ExternalProcedure {
        c_name: cstr!(concatcp!(FN_NAME_PREFIX, "dbl_write")),
        c_fn_ptr: mila_dbl_write as *mut c_void,
        name: "dbl_write",
        param_type: Type::Simple(SimpleType::Double),
        takes_string: false,
    },
    ExternalProcedure {
        c_name: cstr!(concatcp!(FN_NAME_PREFIX, "dbl_writeln")),
        c_fn_ptr: mila_dbl_writeln as *mut c_void,
        name: "dbl_writeln",
        param_type: Type::Simple(SimpleType::Double),
        takes_string: false,
    },
    ExternalProcedure {
        c_name: cstr!(concatcp!(FN_NAME_PREFIX, "str_write")),
        c_fn_ptr: mila_str_write as *mut c_void,
        name: "str_write",
        param_type: Type::Simple(SimpleType::Integer),
        takes_string: true,
    },
    ExternalProcedure {
        c_name: cstr!(concatcp!(FN_NAME_PREFIX, "str_writeln")),
        c_fn_ptr: mila_str_writeln as *mut c_void,
        name: "str_writeln",
        param_type: Type::Simple(SimpleType::Integer),
        takes_string: true,
    },
];

#[no_mangle]
pub extern "C" fn mila_write(int: i64) {
    print!("{}", int);
}

#[no_mangle]
pub extern "C" fn mila_writeln(int: i64) {
    println!("{}", int);
}

#[no_mangle]
pub extern "C" fn mila_dbl_write(double: f64) {
    print!("{}", double);
}

#[no_mangle]
pub extern "C" fn mila_dbl_writeln(double: f64) {
    print!("{}", double);
}

#[no_mangle]
pub extern "C" fn mila_str_write(ptr: *const c_char) {
    unsafe {
        print!("{}", CStr::from_ptr(ptr).to_str().unwrap());
    }
}

#[no_mangle]
pub extern "C" fn mila_str_writeln(ptr: *const c_char) {
    unsafe {
        println!("{}", CStr::from_ptr(ptr).to_str().unwrap());
    }
}
